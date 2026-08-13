import unittest

import torch

from tokenspeed_mla.mla_decode import _get_split_kv_and_workspace_size


def get_split_kv(**overrides) -> int:
    params = {
        "B": 8,
        "q_len": 1,
        "H": 96,
        "kv_lora_rank": 512,
        "max_active_blocks": 148,
        "max_seq_len": 128 * 1024,
        "torch_dtype": torch.float8_e4m3fn,
        "mma_qk_tiler_mn": (128, 128),
    }
    params.update(overrides)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(**params)
    assert workspace_size > 0
    return split_kv


class MLADecodeSplitTest(unittest.TestCase):
    def test_kimi_fp8_m128_uses_two_cta_waves(self) -> None:
        for batch_size, expected_split_kv in (
            (8, 18),
            (16, 9),
            (32, 4),
            (64, 2),
        ):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    get_split_kv(B=batch_size),
                    expected_split_kv,
                )

    def test_two_wave_policy_is_scoped_to_kimi_fp8_m128(self) -> None:
        for override in (
            {"H": 128},
            {"kv_lora_rank": 256},
            {"torch_dtype": torch.bfloat16},
        ):
            with self.subTest(override=override):
                self.assertEqual(get_split_kv(**override), 9)

    def test_two_wave_policy_is_scoped_to_single_token_decode(self) -> None:
        self.assertEqual(get_split_kv(q_len=2), 4)


if __name__ == "__main__":
    unittest.main()
