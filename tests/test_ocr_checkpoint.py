import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from backend.tools.ocr_checkpoint import OcrCheckpointStore
from backend.tools.subtitle_detect import SubtitleDetect


class OcrCheckpointStoreTests(unittest.TestCase):
    def test_incremental_rows_are_merged_and_restored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            store = OcrCheckpointStore(video_path, {"version": 1}, temp_dir)

            store.save(
                300,
                {1: [(10, 110, 20, 50)]},
                {1: [{"box": (10, 110, 20, 50), "text": "高带宽", "score": 0.98}]},
            )
            store.save(600, {400: [(20, 120, 30, 60)]}, complete=True)

            state = store.load()
            self.assertEqual(state.last_frame, 600)
            self.assertTrue(state.complete)
            self.assertEqual(state.sampled_results[1][0], (10, 110, 20, 50))
            self.assertEqual(state.sampled_results[400][0], (20, 120, 30, 60))
            self.assertEqual(state.chinese_records[1][0]["text"], "高带宽")
            self.assertIsInstance(state.chinese_records[1][0]["box"], tuple)

            # A slower concurrent worker must not roll a finished scan back.
            store.save(300, complete=False)
            state = store.load()
            self.assertEqual(state.last_frame, 600)
            self.assertTrue(state.complete)

    def test_changed_fingerprint_invalidates_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            OcrCheckpointStore(video_path, {"version": 1}, temp_dir).save(
                300, {1: [(1, 2, 3, 4)]}
            )

            state = OcrCheckpointStore(
                video_path, {"version": 2}, temp_dir
            ).load()

            self.assertEqual(state.last_frame, 0)
            self.assertFalse(state.complete)
            self.assertEqual(state.sampled_results, {})

    def test_corrupt_database_is_discarded_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            store = OcrCheckpointStore(video_path, {"version": 1}, temp_dir)
            store.path.parent.mkdir(parents=True, exist_ok=True)
            store.path.write_bytes(b"not a sqlite database")

            state = store.load()
            store.save(12, complete=False)

            self.assertEqual(state.last_frame, 0)
            self.assertEqual(store.load().last_frame, 12)


class FakeCapture:
    def __init__(self, frame_count=10, fps=8.0):
        self.frame_count = frame_count
        self.fps = fps
        self.position = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
            return True
        return False

    def isOpened(self):
        return self.position < self.frame_count

    def read(self):
        if not self.isOpened():
            return False, None
        self.position += 1
        return True, object()

    def grab(self):
        if not self.isOpened():
            return False
        self.position += 1
        return True

    def release(self):
        pass


class OcrCheckpointResumeTests(unittest.TestCase):
    def test_failed_scan_resumes_at_first_unfinished_frame_and_reuses_completion(self):
        fake_config = SimpleNamespace(
            removeCjkText=SimpleNamespace(value=False),
            translateNonSubtitleCjk=SimpleNamespace(value=False),
            subtitleDetectMode=SimpleNamespace(value="precise"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")

            with patch(
                "backend.tools.subtitle_detect.cv2.VideoCapture",
                side_effect=lambda *_: FakeCapture(),
            ), patch("backend.tools.subtitle_detect.config", fake_config):
                first = SubtitleDetect(str(video_path), checkpoint_directory=temp_dir)
                first.OCR_CHECKPOINT_INTERVAL_FRAMES = 3
                first_calls = 0

                def fail_on_fifth_frame(_frame):
                    nonlocal first_calls
                    first_calls += 1
                    if first_calls == 5:
                        raise MemoryError("simulated OCR allocation failure")
                    return []

                first.detect_subtitle = fail_on_fifth_frame
                with self.assertRaises(MemoryError):
                    first.find_subtitle_frame_no()

                fingerprint = first._checkpoint_fingerprint(10, None)
                partial = OcrCheckpointStore(
                    video_path, fingerprint, temp_dir
                ).load()
                self.assertEqual(partial.last_frame, 4)
                self.assertFalse(partial.complete)

                second = SubtitleDetect(str(video_path), checkpoint_directory=temp_dir)
                second.OCR_CHECKPOINT_INTERVAL_FRAMES = 3
                resumed_calls = 0

                def finish_remaining_frames(_frame):
                    nonlocal resumed_calls
                    resumed_calls += 1
                    return []

                second.detect_subtitle = finish_remaining_frames
                self.assertEqual(second.find_subtitle_frame_no(), {})
                self.assertEqual(resumed_calls, 6)

                completed = OcrCheckpointStore(
                    video_path, second._checkpoint_fingerprint(10, None), temp_dir
                ).load()
                self.assertEqual(completed.last_frame, 10)
                self.assertTrue(completed.complete)

                third = SubtitleDetect(str(video_path), checkpoint_directory=temp_dir)

                def should_not_run(_frame):
                    raise AssertionError("completed OCR checkpoint was not reused")

                third.detect_subtitle = should_not_run
                self.assertEqual(third.find_subtitle_frame_no(), {})


if __name__ == "__main__":
    unittest.main()
