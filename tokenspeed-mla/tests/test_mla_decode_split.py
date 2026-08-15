import unittest

import torch

from tokenspeed_mla.mla_decode import (
    _get_split_kv_and_workspace_size,
    prepare_mla_decode_ragged_metadata,
)
from tokenspeed_mla.mla_scheduler import (
    plan_ragged_schedule,
    plan_ragged_schedule_and_sanitize_block_tables,
    plan_ragged_splits,
)
from tokenspeed_mla.utils import get_num_sm


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
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_balances_kv_tiles(self) -> None:
        seq_lens = torch.tensor(
            [128, 128, 128, 640],
            dtype=torch.int32,
            device="cuda",
        )
        split_counts = plan_ragged_splits(
            seq_lens,
            target_work_items=8,
            max_splits=8,
        )
        self.assertEqual(split_counts.cpu().tolist(), [1, 1, 1, 5])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_respects_split_cap(self) -> None:
        seq_lens = torch.tensor(
            [128, 4096],
            dtype=torch.int32,
            device="cuda",
        )
        split_counts = plan_ragged_splits(
            seq_lens,
            target_work_items=4,
            max_splits=3,
        )
        self.assertEqual(split_counts.cpu().tolist(), [1, 3])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_rejects_too_small_work_budgets(self) -> None:
        seq_lens = torch.full(
            (4,),
            128,
            dtype=torch.int32,
            device="cuda",
        )
        block_tables = torch.arange(
            4,
            dtype=torch.int32,
            device="cuda",
        ).reshape(4, 1)

        with self.assertRaisesRegex(ValueError, "target_work_items.*batch_size"):
            plan_ragged_schedule(
                seq_lens,
                target_work_items=3,
                max_splits=4,
            )
        with self.assertRaisesRegex(ValueError, "base_work_items.*batch_size"):
            plan_ragged_schedule(
                seq_lens,
                base_work_items=3,
                target_work_items=8,
                max_splits=4,
            )
        with self.assertRaisesRegex(ValueError, "target_work_items.*batch_size"):
            plan_ragged_schedule_and_sanitize_block_tables(
                seq_lens,
                block_tables,
                target_work_items=3,
                max_splits=4,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_materializes_compact_worklist(self) -> None:
        seq_lens = torch.tensor(
            [128, 128, 128, 640],
            dtype=torch.int32,
            device="cuda",
        )
        split_counts, split_offsets, worklist = plan_ragged_schedule(
            seq_lens,
            target_work_items=8,
            max_splits=8,
        )
        self.assertEqual(split_counts.cpu().tolist(), [1, 1, 1, 5])
        self.assertEqual(split_offsets.cpu().tolist(), [0, 1, 2, 3, 8])
        self.assertEqual(
            worklist.cpu().tolist(),
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_marks_unused_workers(self) -> None:
        seq_lens = torch.tensor(
            [128, 128, 128],
            dtype=torch.int32,
            device="cuda",
        )
        split_counts, split_offsets, worklist = plan_ragged_schedule(
            seq_lens,
            target_work_items=8,
            max_splits=8,
        )
        self.assertEqual(split_counts.cpu().tolist(), [1, 1, 1])
        self.assertEqual(split_offsets.cpu().tolist(), [0, 1, 2, 3])
        self.assertEqual(
            worklist.cpu().tolist(),
            [[0, 0], [1, 0], [2, 0]] + [[-1, -1]] * 5,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_planner_fuses_block_table_sanitization(self) -> None:
        seq_lens = torch.tensor(
            [128, 128, 128, 640],
            dtype=torch.int32,
            device="cuda",
        )
        block_tables = torch.tensor(
            [
                [0, -1, 2],
                [-1, 4, 5],
                [6, 7, -1],
                [9, 10, 11],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        sanitized, schedule = (
            plan_ragged_schedule_and_sanitize_block_tables(
                seq_lens,
                block_tables,
                target_work_items=8,
                max_splits=8,
            )
        )
        expected_schedule = plan_ragged_schedule(
            seq_lens,
            target_work_items=8,
            max_splits=8,
        )

        torch.testing.assert_close(
            sanitized,
            torch.where(block_tables == -1, 0, block_tables),
        )
        for actual, expected in zip(schedule, expected_schedule):
            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10,
        "requires Blackwell",
    )
    def test_decode_metadata_helper_matches_k3_schedule_policy(self) -> None:
        seq_lens = torch.arange(
            1,
            9,
            dtype=torch.int32,
            device="cuda",
        ) * 128
        block_tables = torch.tensor(
            [[request_idx, -1] for request_idx in range(8)],
            dtype=torch.int32,
            device="cuda",
        )
        sanitized, schedule = prepare_mla_decode_ragged_metadata(
            seq_lens,
            block_tables,
            max_seq_len=1024,
        )
        split_kv, _ = _get_split_kv_and_workspace_size(
            B=8,
            q_len=1,
            H=96,
            kv_lora_rank=512,
            max_active_blocks=get_num_sm(seq_lens.device),
            max_seq_len=1024,
            torch_dtype=torch.float8_e4m3fn,
            mma_qk_tiler_mn=(128, 128),
        )
        expected_schedule = plan_ragged_schedule(
            seq_lens,
            base_work_items=8 * max(1, split_kv // 2),
            target_work_items=8 * split_kv,
            max_splits=32,
        )

        torch.testing.assert_close(
            sanitized,
            torch.where(block_tables == -1, 0, block_tables),
        )
        for actual, expected in zip(schedule, expected_schedule):
            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_split_planner_scales_between_worker_waves(self) -> None:
        uniform_lens = torch.full(
            (8,),
            7680,
            dtype=torch.int32,
            device="cuda",
        )
        uniform_splits = plan_ragged_splits(
            uniform_lens,
            base_work_items=72,
            target_work_items=144,
            max_splits=32,
        )
        self.assertEqual(uniform_splits.cpu().tolist(), [9] * 8)

        near_uniform_lens = torch.tensor(
            [1280, 1280, 1280, 1408],
            dtype=torch.int32,
            device="cuda",
        )
        near_uniform_splits = plan_ragged_splits(
            near_uniform_lens,
            base_work_items=4,
            target_work_items=8,
            max_splits=8,
        )
        self.assertEqual(near_uniform_splits.cpu().tolist(), [1, 1, 1, 2])

        ragged_lens = torch.tensor(
            [128, 128, 128, 640],
            dtype=torch.int32,
            device="cuda",
        )
        ragged_splits = plan_ragged_splits(
            ragged_lens,
            base_work_items=4,
            target_work_items=8,
            max_splits=8,
        )
        self.assertEqual(ragged_splits.cpu().tolist(), [1, 1, 1, 5])

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
