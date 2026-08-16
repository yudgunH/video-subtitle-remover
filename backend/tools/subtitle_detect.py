import sys
import gc
import os
from difflib import SequenceMatcher
from functools import cached_property

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from .cjk_text import han_character_count, is_chinese_recognition
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections
from backend.tools.ocr_checkpoint import OcrCheckpointStore

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3
    CHINESE_MIN_PERSISTENCE_SAMPLES = 3
    OCR_CHECKPOINT_INTERVAL_FRAMES = 300
    OCR_PIPELINE_VERSION = 4

    def __init__(
        self,
        video_path,
        sub_areas=None,
        checkpoint_directory=None,
        caption_areas=None,
    ):
        self.video_path = video_path
        self.sub_areas = [] if sub_areas is None else sub_areas
        self.caption_areas = [] if caption_areas is None else caption_areas
        self.checkpoint_directory = checkpoint_directory
        self.last_cjk_records = []
        self.cjk_text_by_frame = {}
        self._init_sample_step()

    def _init_sample_step(self):
        """Sample roughly eight OCR frames/sec for stable Chinese text."""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.video_fps = float(fps or 0.0)
        if self.video_fps <= 0:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = max(1, min(8, int(round(self.video_fps / 8.0))))

    @staticmethod
    def _serialize_sections(sections):
        if not sections:
            return None
        serialized = []
        for section in sections:
            if isinstance(section, range):
                serialized.append([section.start, section.stop, section.step])
            else:
                serialized.append(list(section))
        return serialized

    def _checkpoint_fingerprint(self, frame_count, sub_remover):
        readable_path = get_readable_path(self.video_path)
        stat = os.stat(readable_path)
        detect_mode = config.subtitleDetectMode.value
        return {
            "pipeline_version": self.OCR_PIPELINE_VERSION,
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "frame_count": int(frame_count),
            "fps": round(self.video_fps, 6),
            "sample_step": self.SAMPLE_STEP,
            "detect_mode": getattr(detect_mode, "name", str(detect_mode)),
            "chinese_only": bool(
                config.removeCjkText.value or config.translateNonSubtitleCjk.value
            ),
            "sub_areas": [list(area) for area in self.sub_areas],
            "caption_areas": [list(area) for area in self.caption_areas],
            "ab_sections": self._serialize_sections(
                getattr(sub_remover, "ab_sections", None)
            ),
            "minimum_persistence": self.CHINESE_MIN_PERSISTENCE_SAMPLES,
        }

    @cached_property
    def paddle_device(self):
        """Select Paddle GPU 0 when both app acceleration and CUDA Paddle exist."""
        import paddle

        hardware_accelerator = HardwareAccelerator.instance()
        if not hardware_accelerator.has_cuda():
            return "cpu"
        try:
            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                return "gpu:0"
        except Exception:
            pass
        print("CUDA is available to Torch, but PaddleOCR has no CUDA runtime; using CPU OCR.")
        return "cpu"

    @staticmethod
    def _create_ocr_model(model_class, device, **kwargs):
        """Create an OCR model on GPU and gracefully fall back to CPU."""
        try:
            model = model_class(device=device, **kwargs)
            return model, device
        except Exception:
            if device == "cpu":
                raise
            print(f"Failed to initialize {model_class.__name__} on {device}; using CPU.")
            model = model_class(device="cpu", **kwargs)
            return model, "cpu"

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        model_config = ModelConfig()
        model, actual_device = self._create_ocr_model(
            TextDetection,
            self.paddle_device,
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            enable_hpi=len(onnx_providers) > 0,
        )
        print(f"PaddleOCR text detection device: {actual_device}")
        return model

    @cached_property
    def chinese_text_recognizer(self):
        """Load only the Chinese recognizer used by Chinese-only mode."""
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextRecognition

        hardware_accelerator = HardwareAccelerator.instance()
        enable_hpi = len(hardware_accelerator.onnx_providers) > 0
        model_config = ModelConfig()
        recognizer, actual_device = self._create_ocr_model(
            TextRecognition,
            self.paddle_device,
            model_name=model_config.REC_MODEL_NAME,
            enable_hpi=enable_hpi,
        )
        print(f"PaddleOCR Chinese recognition device: {actual_device}")
        return recognizer

    def release_models(self):
        """Release Paddle models before the Torch inpainting phase uses VRAM."""
        self.__dict__.pop("text_detector", None)
        self.__dict__.pop("chinese_text_recognizer", None)
        gc.collect()
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _crop_text_region(img, coordinate):
        xmin, xmax, ymin, ymax = coordinate
        height, width = img.shape[:2]
        xmin = max(0, min(int(xmin), width))
        xmax = max(0, min(int(xmax), width))
        ymin = max(0, min(int(ymin), height))
        ymax = max(0, min(int(ymax), height))
        if xmax <= xmin or ymax <= ymin:
            return None
        return img[ymin:ymax, xmin:xmax]

    @staticmethod
    def _recognize_batch(recognizer, crops):
        if not crops:
            return []
        results = recognizer.predict(input=crops, batch_size=min(len(crops), 16))
        recognized = []
        for result in results:
            try:
                recognized.append((result['rec_text'], result['rec_score']))
            except (KeyError, TypeError):
                recognized.append(("", 0.0))
        return recognized

    @staticmethod
    def _is_plausible_chinese_box(coordinate, text):
        """Reject tiny or vertical shapes before they can become erase masks."""
        xmin, xmax, ymin, ymax = coordinate
        width = xmax - xmin
        height = ymax - ymin
        han_count = han_character_count(text)
        if width < 24 or height < 10 or width * height < 320:
            return False
        if width <= height:
            return False
        # A line recognized as several Han characters needs enough horizontal
        # image evidence; this rejects component edges read as a long phrase.
        return width >= max(24, han_count * 5)

    def _recognize_chinese_records(self, img, coordinates):
        """Return high-confidence Chinese OCR records from one frame."""
        valid_coordinates = []
        crops = []
        for coordinate in coordinates:
            crop = self._crop_text_region(img, coordinate)
            if crop is None or crop.size == 0:
                continue
            valid_coordinates.append(coordinate)
            crops.append(crop)

        if not crops:
            return []

        recognition_results = self._recognize_batch(
            self.chinese_text_recognizer, crops
        )
        matched = []
        for index, (text, score) in enumerate(recognition_results):
            if index >= len(valid_coordinates):
                break
            coordinate = valid_coordinates[index]
            if not is_chinese_recognition(text, score):
                continue
            if not self._is_plausible_chinese_box(coordinate, text):
                continue
            matched.append({
                "box": tuple(coordinate),
                "text": str(text).strip(),
                "score": float(score),
            })
        return matched

    def _filter_cjk_coordinates(self, img, coordinates):
        """Compatibility wrapper returning conservative Chinese boxes."""
        return [record["box"] for record in self._recognize_chinese_records(img, coordinates)]

    @staticmethod
    def _box_center_in_areas(coordinate, areas):
        """Use box center so glyphs touching a caption boundary are retained."""
        xmin, xmax, ymin, ymax = coordinate
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        return any(
            area_ymin <= center_y <= area_ymax
            and area_xmin <= center_x <= area_xmax
            for area_ymin, area_ymax, area_xmin, area_xmax in (areas or [])
        )

    @staticmethod
    def _box_iou(first, second):
        ax1, ax2, ay1, ay2 = first
        bx1, bx2, by1, by2 = second
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0, min(ay2, by2) - max(ay1, by1)
        )
        if intersection <= 0:
            return 0.0
        first_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        second_area = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / float(first_area + second_area - intersection)

    @staticmethod
    def _han_only(text):
        return "".join(
            character for character in str(text)
            if han_character_count(character)
        )

    @classmethod
    def _same_chinese_candidate(cls, first, second):
        if cls._box_iou(first["box"], second["box"]) < 0.35:
            return False
        first_text = cls._han_only(first["text"])
        second_text = cls._han_only(second["text"])
        if not first_text or not second_text:
            return False
        if first_text in second_text or second_text in first_text:
            return True
        return SequenceMatcher(None, first_text, second_text).ratio() >= 0.65

    def _filter_stable_chinese_records(self, records_by_frame):
        """Keep only Chinese boxes confirmed across multiple sampled frames."""
        tracks = []
        max_gap = max(1, self.SAMPLE_STEP * 2)
        for frame_no in sorted(records_by_frame):
            used_tracks = set()
            for record in records_by_frame[frame_no]:
                best_index = None
                best_iou = 0.0
                for index, track in enumerate(tracks):
                    if index in used_tracks or frame_no - track["last_frame"] > max_gap:
                        continue
                    if not self._same_chinese_candidate(track["last_record"], record):
                        continue
                    iou = self._box_iou(track["last_record"]["box"], record["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_index is None:
                    tracks.append({
                        "last_frame": frame_no,
                        "last_record": record,
                        "detections": [(frame_no, record)],
                    })
                    used_tracks.add(len(tracks) - 1)
                else:
                    track = tracks[best_index]
                    track["last_frame"] = frame_no
                    track["last_record"] = record
                    track["detections"].append((frame_no, record))
                    used_tracks.add(best_index)

        stable = {}
        for track in tracks:
            distinct_frames = {frame_no for frame_no, _ in track["detections"]}
            if len(distinct_frames) < self.CHINESE_MIN_PERSISTENCE_SAMPLES:
                continue
            for frame_no, record in track["detections"]:
                stable.setdefault(frame_no, []).append(record)
        return stable

    def detect_subtitle(self, img):
        temp_list = []
        self.last_cjk_records = []
        results = self.text_detector.predict(img)
        sub_areas = self.sub_areas
        remove_cjk_text = (
            config.removeCjkText.value or config.translateNonSubtitleCjk.value
        )
        has_areas = sub_areas is not None and len(sub_areas) > 0
        for res in results:
            dt_polys = res['dt_polys']
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list = get_coordinates(dt_polys.tolist())
            if not coordinate_list:
                continue
            if remove_cjk_text:
                # Hybrid mode: every detected text box in a user-trusted caption
                # area is removed. Elsewhere, only conservatively confirmed
                # Chinese text is allowed through.
                caption_coordinates = [
                    coordinate for coordinate in coordinate_list
                    if self._box_center_in_areas(coordinate, self.caption_areas)
                ]
                outside_coordinates = [
                    coordinate for coordinate in coordinate_list
                    if not self._box_center_in_areas(
                        coordinate, self.caption_areas
                    )
                ]
                records = self._recognize_chinese_records(
                    img, outside_coordinates
                )
                self.last_cjk_records.extend(records)
                temp_list.extend(caption_coordinates)
                temp_list.extend(record["box"] for record in records)
            elif not has_areas:
                temp_list.extend(coordinate_list)
            elif len(sub_areas) == 1:
                # 单区域快速路径（最常见场景）
                s_ymin, s_ymax, s_xmin, s_xmax = sub_areas[0]
                for xmin, xmax, ymin, ymax in coordinate_list:
                    if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                        temp_list.append((xmin, xmax, ymin, ymax))
            else:
                for xmin, xmax, ymin, ymax in coordinate_list:
                    for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
                        if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                            temp_list.append((xmin, xmax, ymin, ymax))
                            break
        return temp_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        checkpoint = OcrCheckpointStore(
            self.video_path,
            self._checkpoint_fingerprint(frame_count, sub_remover),
            self.checkpoint_directory,
        )
        state = checkpoint.load()
        current_frame_no = min(max(0, state.last_frame), frame_count)
        sampled_results = dict(state.sampled_results)
        self.cjk_text_by_frame = {}
        raw_chinese_records = dict(state.chinese_records)
        chinese_only_mode = (
            config.removeCjkText.value or config.translateNonSubtitleCjk.value
        )
        tbar = tqdm(
            total=frame_count,
            initial=frame_count if state.complete else current_frame_no,
            unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding'
        )
        if sub_remover:
            if state.complete:
                sub_remover.progress_total = 50
                sub_remover.append_output(
                    tr['Main']['OcrCheckpointComplete'].format(frame_count)
                )
            elif current_frame_no > 0:
                percent = 100.0 * current_frame_no / max(1, frame_count)
                sub_remover.progress_total = percent // 2
                sub_remover.append_output(
                    tr['Main']['OcrCheckpointResumed'].format(
                        current_frame_no, percent
                    )
                )
            else:
                sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
            if hasattr(sub_remover, "notify_progress_listeners"):
                sub_remover.notify_progress_listeners()

        pending_results = {}
        pending_records = {}
        last_completed_frame = current_frame_no
        last_checkpoint_frame = current_frame_no
        sections = getattr(sub_remover, "ab_sections", None)
        if not state.complete and current_frame_no:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no)

        try:
            while not state.complete and video_cap.isOpened():
                zero_based_frame = current_frame_no
                in_section = is_frame_number_in_ab_sections(
                    zero_based_frame, sections
                )
                should_ocr = in_section and (
                    zero_based_frame % self.SAMPLE_STEP == 0
                    or self.SAMPLE_STEP <= 1
                )
                if should_ocr:
                    ret, frame = video_cap.read()
                else:
                    ret = video_cap.grab()
                    frame = None
                if not ret:
                    break

                current_frame_no += 1
                if should_ocr:
                    temp_list = self.detect_subtitle(frame)
                    if temp_list:
                        sampled_results[current_frame_no] = temp_list
                        pending_results[current_frame_no] = temp_list
                        if self.last_cjk_records:
                            records = [
                                dict(record) for record in self.last_cjk_records
                            ]
                            raw_chinese_records[current_frame_no] = records
                            pending_records[current_frame_no] = records

                last_completed_frame = current_frame_no
                tbar.update(1)
                if sub_remover:
                    sub_remover.progress_total = (
                        100 * float(current_frame_no) / max(1.0, float(frame_count))
                    ) // 2

                if (
                    last_completed_frame - last_checkpoint_frame
                    >= self.OCR_CHECKPOINT_INTERVAL_FRAMES
                ):
                    checkpoint.save(
                        last_completed_frame,
                        pending_results,
                        pending_records,
                        complete=False,
                    )
                    pending_results.clear()
                    pending_records.clear()
                    last_checkpoint_frame = last_completed_frame
        except BaseException:
            checkpoint.save(
                last_completed_frame,
                pending_results,
                pending_records,
                complete=False,
            )
            raise
        else:
            if not state.complete:
                checkpoint.save(
                    last_completed_frame,
                    pending_results,
                    pending_records,
                    complete=True,
                )
        finally:
            video_cap.release()
            tbar.close()

        if chinese_only_mode:
            self.cjk_text_by_frame = self._filter_stable_chinese_records(
                raw_chinese_records
            )
            caption_results = {
                frame_no: [
                    box for box in boxes
                    if self._box_center_in_areas(box, self.caption_areas)
                ]
                for frame_no, boxes in sampled_results.items()
            }
            merged_results = {}
            for frame_no in set(caption_results).union(self.cjk_text_by_frame):
                boxes = list(caption_results.get(frame_no, []))
                for record in self.cjk_text_by_frame.get(frame_no, []):
                    box = record["box"]
                    if box not in boxes:
                        boxes.append(box)
                if boxes:
                    merged_results[frame_no] = boxes
            sampled_results = merged_results
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧也标记为有字幕
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                fill_mask = sampled_results[f]
                for fill_f in range(f + 1, next_f):
                    subtitle_frame_no_box_dict[fill_f] = fill_mask
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
