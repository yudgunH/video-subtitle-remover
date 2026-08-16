import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.tools.cjk_text import (
    contains_han,
    is_chinese_recognition,
)
from backend.tools.args_handler import parse_args
from backend.tools.ocr import get_coordinates
from backend.tools.subtitle_detect import SubtitleDetect
from backend.main import SubtitleRemover


class ChineseTextTests(unittest.TestCase):
    def test_detects_chinese(self):
        self.assertTrue(contains_han("第12集 Episode 12"))

    def test_does_not_treat_japanese_kana_as_chinese(self):
        self.assertFalse(is_chinese_recognition("メモリ帯域", 0.99))

    def test_does_not_treat_korean_as_chinese(self):
        self.assertFalse(is_chinese_recognition("가격 10,000원", 0.99))

    def test_detects_supplementary_han(self):
        self.assertTrue(contains_han("𠀀"))

    def test_keeps_latin_and_numbers(self):
        self.assertFalse(contains_han("Episode 12 - price: 10,000"))

    def test_rejects_low_confidence_chinese(self):
        self.assertFalse(is_chinese_recognition("中文字幕", 0.84))

    def test_rejects_single_han_character(self):
        self.assertFalse(is_chinese_recognition("中", 0.99))

    def test_accepts_confident_chinese_phrase(self):
        self.assertTrue(is_chinese_recognition("中文字幕", 0.99))

    def test_rejects_tiny_text_like_box(self):
        self.assertFalse(
            SubtitleDetect._is_plausible_chinese_box((10, 21, 10, 19), "中国")
        )

    def test_requires_three_consistent_samples(self):
        detector = SubtitleDetect.__new__(SubtitleDetect)
        detector.SAMPLE_STEP = 2
        stable = detector._filter_stable_chinese_records({
            1: [
                {"box": (100, 220, 50, 80), "text": "高带宽", "score": 0.99},
                {"box": (20, 60, 20, 40), "text": "误判", "score": 0.91},
            ],
            3: [{"box": (102, 222, 51, 81), "text": "高带宽", "score": 0.99}],
            5: [{"box": (104, 224, 52, 82), "text": "高带宽存储", "score": 0.98}],
        })
        self.assertEqual(sorted(stable), [1, 3, 5])
        self.assertTrue(all(records[0]["text"].startswith("高带宽") for records in stable.values()))

    def test_rejects_inconsistent_persistent_shapes(self):
        detector = SubtitleDetect.__new__(SubtitleDetect)
        detector.SAMPLE_STEP = 2
        stable = detector._filter_stable_chinese_records({
            1: [{"box": (100, 220, 50, 80), "text": "电路", "score": 0.90}],
            3: [{"box": (101, 221, 50, 80), "text": "显卡", "score": 0.91}],
            5: [{"box": (102, 222, 50, 80), "text": "芯片", "score": 0.92}],
        })
        self.assertEqual(stable, {})

    def test_rotated_polygon_uses_full_outer_bounds(self):
        polygon = [[[10, 20], [90, 10], [100, 50], [20, 60]]]
        self.assertEqual(get_coordinates(polygon), [(10, 100, 10, 60)])

    def test_cli_enables_cjk_removal(self):
        with patch("sys.argv", ["vsr", "-i", "video.mp4", "--remove-cjk-text"]):
            args = parse_args()
        self.assertTrue(args.remove_cjk_text)
        self.assertIsNone(args.output)

    def test_cli_translation_uses_secret_from_environment(self):
        with patch(
            "sys.argv",
            [
                "vsr", "-i", "video.mp4",
                "--translate-non-subtitle-cjk",
                "--translation-target-language", "Vietnamese",
                "--nine-router-model", "auto",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.translate_non_subtitle_cjk)
        self.assertEqual(args.translation_target_language, "Vietnamese")
        self.assertFalse(hasattr(args, "nine_router_api_key"))

    def test_cjk_mode_forces_full_frame_over_saved_subtitle_area(self):
        areas = SubtitleRemover.resolve_processing_areas(
            [(634, 713, 192, 1088)],
            frame_height=720,
            frame_width=1280,
            remove_cjk_text=True,
        )
        self.assertEqual(areas, [(0, 720, 0, 1280)])

    def test_hybrid_detection_keeps_all_caption_text_but_filters_outside(self):
        detector = SubtitleDetect.__new__(SubtitleDetect)
        detector.sub_areas = [(0, 120, 0, 200)]
        detector.caption_areas = [(80, 120, 0, 200)]
        detector.last_cjk_records = []
        caption_box = (20, 150, 88, 112)
        outside_box = (30, 140, 20, 45)
        polygons = np.array([
            [[20, 88], [150, 88], [150, 112], [20, 112]],
            [[30, 20], [140, 20], [140, 45], [30, 45]],
        ])
        detector.__dict__["text_detector"] = SimpleNamespace(
            predict=lambda _img: [{"dt_polys": polygons}]
        )

        def recognize_only_outside(_img, coordinates):
            self.assertEqual(coordinates, [outside_box])
            return [{
                "box": outside_box,
                "text": "高带宽",
                "score": 0.99,
            }]

        detector._recognize_chinese_records = recognize_only_outside
        fake_config = SimpleNamespace(
            removeCjkText=SimpleNamespace(value=True),
            translateNonSubtitleCjk=SimpleNamespace(value=False),
        )
        with patch("backend.tools.subtitle_detect.config", fake_config):
            boxes = detector.detect_subtitle(
                np.zeros((120, 200, 3), dtype=np.uint8)
            )

        self.assertEqual(boxes, [caption_box, outside_box])
        self.assertEqual(
            [record["box"] for record in detector.last_cjk_records],
            [outside_box],
        )


if __name__ == "__main__":
    unittest.main()
