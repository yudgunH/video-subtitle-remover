import unittest

import cv2
import numpy as np

from backend.tools.inpaint_tools import (
    create_feathered_mask_alpha,
    create_caption_text_union_mask,
    create_chinese_text_mask,
    create_chinese_text_union_mask,
    create_text_box_mask,
)


class ChineseTextMaskTests(unittest.TestCase):
    def test_masks_neutral_text_strokes_but_preserves_colored_graphic(self):
        frame = np.full((100, 240, 3), (35, 45, 35), dtype=np.uint8)
        box = (40, 200, 25, 80)
        cv2.line(frame, (45, 70), (195, 30), (0, 0, 255), 5)
        # Synthetic neutral strokes spread like a short Chinese phrase.
        for x in (60, 95, 130, 165):
            cv2.rectangle(frame, (x, 38), (x + 18, 65), (245, 245, 245), 3)
            cv2.line(frame, (x, 51), (x + 18, 51), (245, 245, 245), 3)

        mask = create_chinese_text_mask(frame, [box])

        self.assertGreater(np.count_nonzero(mask[35:68, 55:190]), 100)
        # Red-only part of the graphic lies inside the OCR box but must not be masked.
        self.assertEqual(np.count_nonzero(mask[68:75, 44:52]), 0)
        self.assertEqual(np.count_nonzero(mask[:20]), 0)

    def test_skips_ambiguous_tiny_candidate(self):
        frame = np.full((80, 160, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (80, 40), 3, (255, 255, 255), -1)
        mask = create_chinese_text_mask(frame, [(30, 130, 20, 60)])
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_mask_extends_past_tight_ocr_box_for_antialiased_edges(self):
        frame = np.full((80, 180, 3), 35, dtype=np.uint8)
        cv2.putText(
            frame, "TEST", (35, 53), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (245, 245, 245), 2, cv2.LINE_AA,
        )
        tight_box = (40, 125, 28, 57)

        mask = create_chinese_text_mask(frame, [tight_box])

        self.assertGreater(np.count_nonzero(mask[26:59, 38:127]), 100)
        self.assertGreater(np.count_nonzero(mask[:, :40]), 0)
        self.assertEqual(np.count_nonzero(mask[:20]), 0)

    def test_temporal_union_keeps_shifted_text_positions(self):
        first = np.full((90, 220, 3), 35, dtype=np.uint8)
        second = first.copy()
        cv2.putText(first, "TEXT", (50, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 245, 245), 2)
        cv2.putText(second, "TEXT", (58, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 245, 245), 2)

        first_mask = create_chinese_text_mask(first, [(45, 145, 30, 65)])
        union = create_chinese_text_union_mask([
            (first, [(45, 145, 30, 65)]),
            (second, [(53, 153, 30, 65)]),
        ])

        self.assertGreater(np.count_nonzero(union), np.count_nonzero(first_mask))
        self.assertGreater(np.count_nonzero(union[:, 129:140]), 0)

    def test_feather_keeps_hard_core_and_only_softens_nearby_pixels(self):
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[30:50, 45:75] = 255

        alpha = create_feathered_mask_alpha(mask, radius=3)

        self.assertEqual(float(alpha[40, 60]), 1.0)
        self.assertGreater(float(alpha[29, 60]), 0.0)
        self.assertLess(float(alpha[27, 60]), 1.0)
        self.assertEqual(float(alpha[10, 10]), 0.0)

    def test_caption_box_mask_covers_whole_detection_with_small_padding(self):
        mask = create_text_box_mask(
            (80, 120), [(30, 90, 35, 50)], padding=3
        )

        self.assertTrue(np.all(mask[32:54, 27:94] == 255))
        self.assertEqual(np.count_nonzero(mask[:25]), 0)

    def test_partial_caption_box_expands_to_the_complete_visual_line(self):
        frame = np.full((100, 380, 3), 25, dtype=np.uint8)
        cv2.putText(
            frame, "GRIP STRENGTH", (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245),
            2, cv2.LINE_AA,
        )
        # A separate bright object must not pull the line mask to the edge.
        cv2.rectangle(frame, (345, 25), (355, 75), (245, 245, 245), -1)
        partial_ocr_box = (120, 195, 38, 70)

        mask = create_caption_text_union_mask(
            [(frame, [partial_ocr_box])],
            [(20, 80, 0, 370)],
            size=frame.shape[:2],
        )

        self.assertGreater(np.count_nonzero(mask[35:75, 15:80]), 0)
        self.assertGreater(np.count_nonzero(mask[35:75, 230:300]), 0)
        self.assertEqual(np.count_nonzero(mask[:, 340:365]), 0)

    def test_caption_margin_remains_wide_when_projection_only_finds_middle(self):
        frame = np.full((100, 400, 3), 25, dtype=np.uint8)
        # Only the central OCR fragment has visual text evidence. The missing
        # end fragments still need to fall inside the deterministic line mask.
        cv2.putText(
            frame, "MIDDLE", (155, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (245, 245, 245), 2, cv2.LINE_AA,
        )
        partial_ocr_box = (160, 240, 40, 70)

        mask = create_caption_text_union_mask(
            [(frame, [partial_ocr_box])],
            [(20, 80, 20, 380)],
            size=frame.shape[:2],
        )

        self.assertGreater(np.count_nonzero(mask[35:75, 55:85]), 0)
        self.assertGreater(np.count_nonzero(mask[35:75, 315:345]), 0)
        self.assertEqual(np.count_nonzero(mask[:, :15]), 0)


if __name__ == "__main__":
    unittest.main()
