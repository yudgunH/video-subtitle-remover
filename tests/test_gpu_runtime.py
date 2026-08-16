import os
import unittest
from unittest.mock import patch

import torch

from backend.tools.torch_inference import cuda_autocast
from backend.inpaint.sttn.network_sttn import Attention


class GpuRuntimeTests(unittest.TestCase):
    def test_cpu_uses_noop_context(self):
        with cuda_autocast(torch.device("cpu")):
            value = torch.tensor([1.0]) + 1
        self.assertEqual(value.item(), 2.0)

    def test_fp16_can_be_disabled(self):
        with patch.dict(os.environ, {"VSR_DISABLE_FP16": "1"}):
            with cuda_autocast(torch.device("cuda:0")):
                value = 1
        self.assertEqual(value, 1)

    def test_sttn_attention_mask_supports_fp16(self):
        query = torch.ones((1, 1, 2), dtype=torch.float16)
        key = torch.ones((1, 2, 2), dtype=torch.float16)
        value = torch.tensor([[[1.0], [9.0]]], dtype=torch.float16)
        mask = torch.tensor([[[True, False]]])

        output, weights = Attention()(query, key, value, mask)

        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(output.item(), 9.0)


if __name__ == "__main__":
    unittest.main()
