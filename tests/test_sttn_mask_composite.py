import unittest

import cv2
import numpy as np

from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


class SttnMaskCompositeTests(unittest.TestCase):
    def test_pixels_outside_mask_are_bit_identical(self):
        inpainter = STTNDetInpaint.__new__(STTNDetInpaint)
        inpainter.model_input_width = 432
        inpainter.model_input_height = 240
        inpainter.inpaint = lambda frames, masks: [
            np.full_like(frame, (20, 80, 200)) for frame in frames
        ]

        original = np.arange(72 * 128 * 3, dtype=np.uint8).reshape(72, 128, 3)
        mask = np.zeros((72, 128), dtype=np.uint8)
        mask[30:42, 50:82] = 255

        result = inpainter([original], mask)[0]
        outside = mask == 0
        inside = mask > 0

        self.assertTrue(np.array_equal(result[outside], original[outside]))
        self.assertFalse(np.array_equal(result[inside], original[inside]))

    def test_wide_model_mask_does_not_expand_final_composite(self):
        inpainter = STTNDetInpaint.__new__(STTNDetInpaint)
        inpainter.model_input_width = 432
        inpainter.model_input_height = 240
        inpainter.inpaint = lambda frames, masks: [
            np.full_like(frame, (10, 30, 220)) for frame in frames
        ]

        original = np.full((72, 128, 3), 90, dtype=np.uint8)
        model_mask = np.zeros((72, 128), dtype=np.uint8)
        model_mask[20:52, 30:100] = 255
        stroke_mask = np.zeros((72, 128), dtype=np.uint8)
        stroke_mask[30:42, 50:82] = 255

        result = inpainter(
            [original], model_mask, composite_mask=stroke_mask
        )[0]

        protected = cv2.dilate(
            stroke_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        far_from_text = (model_mask > 0) & (protected == 0)
        halo = (protected > 0) & (stroke_mask == 0)

        self.assertTrue(np.array_equal(result[far_from_text], original[far_from_text]))
        self.assertFalse(np.array_equal(result[stroke_mask > 0], original[stroke_mask > 0]))
        self.assertFalse(np.array_equal(result[halo], original[halo]))


if __name__ == "__main__":
    unittest.main()
