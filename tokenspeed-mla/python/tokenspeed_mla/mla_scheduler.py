# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GPU-side work planning for ragged MLA decode."""

import torch

from tokenspeed_mla._triton import tl, triton


@triton.jit
def _plan_ragged_splits_kernel(
    seq_lens_ptr,
    split_counts_ptr,
    split_offsets_ptr,
    worklist_ptr,
    block_tables_ptr,
    sanitized_block_tables_ptr,
    block_table_elements,
    batch_size,
    base_work_items,
    target_work_items,
    max_splits,
    KV_TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    WORK_BLOCK_SIZE: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
    SANITIZE_BLOCK_TABLES: tl.constexpr,
):
    program_idx = tl.program_id(axis=0)
    if SANITIZE_BLOCK_TABLES:
        copy_offsets = program_idx * COPY_BLOCK_SIZE + tl.arange(
            0, COPY_BLOCK_SIZE
        )
        copy_mask = copy_offsets < block_table_elements
        block_ids = tl.load(
            block_tables_ptr + copy_offsets,
            mask=copy_mask,
            other=0,
        )
        block_ids = tl.where(block_ids == -1, 0, block_ids)
        tl.store(
            sanitized_block_tables_ptr + copy_offsets,
            block_ids,
            mask=copy_mask,
        )

    # Only one program builds the batch-wide schedule. Other programs only
    # participate in the block-table copy above.
    if program_idx != 0:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    seq_lens = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    k_tiles = (seq_lens + KV_TILE_SIZE - 1) // KV_TILE_SIZE
    total_k_tiles = tl.sum(k_tiles, axis=0)
    max_k_tiles = tl.max(k_tiles, axis=0)
    safe_total_k_tiles = tl.maximum(total_k_tiles, 1)
    excess_k_tiles = tl.maximum(
        max_k_tiles * batch_size - total_k_tiles,
        0,
    )
    imbalance_fraction_numerator = tl.minimum(
        excess_k_tiles,
        total_k_tiles,
    )
    extra_work_capacity = target_work_items - base_work_items
    effective_work_items = base_work_items + (
        extra_work_capacity * imbalance_fraction_numerator
        + safe_total_k_tiles
        - 1
    ) // safe_total_k_tiles
    target_payload = tl.maximum(
        1,
        (total_k_tiles + effective_work_items - 1) // effective_work_items,
    )
    split_counts = tl.maximum(
        1,
        (k_tiles + target_payload - 1) // target_payload,
    )
    split_counts = tl.minimum(split_counts, max_splits)
    split_counts = tl.where(mask, split_counts, 0)
    planned_work_items = tl.sum(split_counts, axis=0)
    while planned_work_items > effective_work_items:
        target_payload += 1
        split_counts = tl.maximum(
            1,
            (k_tiles + target_payload - 1) // target_payload,
        )
        split_counts = tl.minimum(split_counts, max_splits)
        split_counts = tl.where(mask, split_counts, 0)
        planned_work_items = tl.sum(split_counts, axis=0)

    inclusive_offsets = tl.cumsum(split_counts, axis=0)
    split_offsets = inclusive_offsets - split_counts
    tl.store(split_counts_ptr + offsets, split_counts, mask=mask)
    tl.store(split_offsets_ptr + offsets, split_offsets, mask=mask)
    tl.store(split_offsets_ptr + batch_size, planned_work_items)
    tl.debug_barrier()

    work_idx = tl.arange(0, WORK_BLOCK_SIZE)
    work_mask = work_idx < target_work_items
    work_batch = work_idx * 0 - 1
    work_split = work_idx * 0 - 1
    for request_idx in tl.static_range(0, BATCH_SIZE):
        request_offset = tl.load(split_offsets_ptr + request_idx)
        request_splits = tl.load(split_counts_ptr + request_idx)
        is_request_work = (
            (work_idx >= request_offset)
            & (work_idx < request_offset + request_splits)
        )
        work_batch = tl.where(is_request_work, request_idx, work_batch)
        work_split = tl.where(
            is_request_work,
            work_idx - request_offset,
            work_split,
        )
    tl.store(worklist_ptr + work_idx * 2, work_batch, mask=work_mask)
    tl.store(worklist_ptr + work_idx * 2 + 1, work_split, mask=work_mask)


def plan_ragged_schedule(
    seq_lens: torch.Tensor,
    *,
    target_work_items: int,
    base_work_items: int | None = None,
    max_splits: int,
    kv_tile_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build split counts, prefix offsets, and an O(1) compact worklist.

    The useful-work budget grows linearly from ``base_work_items`` at uniform
    lengths to ``target_work_items`` once the longest request is at least twice
    the mean. Launch capacity remains fixed at ``target_work_items``.
    """
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1-D, got shape={tuple(seq_lens.shape)}")
    if seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got dtype={seq_lens.dtype}")
    if target_work_items <= 0:
        raise ValueError(
            f"target_work_items must be positive, got {target_work_items}"
        )
    if base_work_items is None:
        base_work_items = target_work_items
    if not 0 < base_work_items <= target_work_items:
        raise ValueError(
            "base_work_items must be in [1, target_work_items], got "
            f"{base_work_items} for target_work_items={target_work_items}"
        )
    if max_splits <= 0:
        raise ValueError(f"max_splits must be positive, got {max_splits}")
    if kv_tile_size <= 0:
        raise ValueError(f"kv_tile_size must be positive, got {kv_tile_size}")

    batch_size = seq_lens.numel()
    if target_work_items < batch_size:
        raise ValueError(
            "target_work_items must provide at least one slot per request, got "
            f"{target_work_items} for batch_size={batch_size}"
        )
    if base_work_items < batch_size:
        raise ValueError(
            "base_work_items must be at least batch_size because every request "
            f"requires one split, got {base_work_items} for batch_size={batch_size}"
        )
    if batch_size == 0:
        empty = torch.empty_like(seq_lens)
        return (
            empty,
            torch.zeros(1, dtype=torch.int32, device=seq_lens.device),
            torch.full(
                (target_work_items, 2),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            ),
        )

    block_size = triton.next_power_of_2(batch_size)
    if block_size > 1024:
        raise ValueError(
            f"ragged scheduler supports at most 1024 requests, got {batch_size}"
        )
    work_block_size = triton.next_power_of_2(target_work_items)
    if work_block_size > 65536:
        raise ValueError(
            "ragged scheduler supports at most 65536 work items, "
            f"got {target_work_items}"
        )

    split_counts = torch.empty_like(seq_lens)
    split_offsets = torch.empty(
        batch_size + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    worklist = torch.empty(
        (target_work_items, 2),
        dtype=torch.int32,
        device=seq_lens.device,
    )
    _plan_ragged_splits_kernel[(1,)](
        seq_lens,
        split_counts,
        split_offsets,
        worklist,
        seq_lens,
        seq_lens,
        0,
        batch_size,
        base_work_items,
        target_work_items,
        max_splits,
        KV_TILE_SIZE=kv_tile_size,
        BLOCK_SIZE=block_size,
        BATCH_SIZE=batch_size,
        WORK_BLOCK_SIZE=work_block_size,
        COPY_BLOCK_SIZE=1,
        SANITIZE_BLOCK_TABLES=False,
    )
    return split_counts, split_offsets, worklist


def plan_ragged_schedule_and_sanitize_block_tables(
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    target_work_items: int,
    base_work_items: int | None = None,
    max_splits: int,
    kv_tile_size: int = 128,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Sanitize page IDs and build the ragged schedule in one GPU launch.

    This replaces the serving path's existing ``block_tables == -1`` copy and
    emits planner metadata from program 0 of the same launch. The schedule
    extent stays fixed at ``target_work_items`` for CUDA graph replay.
    """
    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must be 2-D, got "
            f"shape={tuple(block_tables.shape)}"
        )
    if block_tables.dtype != torch.int32:
        raise ValueError(
            f"block_tables must be int32, got dtype={block_tables.dtype}"
        )
    if block_tables.device != seq_lens.device:
        raise ValueError(
            "block_tables and seq_lens must be on the same device, got "
            f"{block_tables.device} and {seq_lens.device}"
        )
    if block_tables.shape[0] != seq_lens.numel():
        raise ValueError(
            "block_tables rows must match seq_lens, got "
            f"{block_tables.shape[0]} and {seq_lens.numel()}"
        )

    # Reuse the public planner's validation and empty-batch behavior without
    # launching it. Empty serving batches do not carry block-table rows.
    if seq_lens.numel() == 0:
        sanitized = torch.empty_like(block_tables)
        schedule = plan_ragged_schedule(
            seq_lens,
            target_work_items=target_work_items,
            base_work_items=base_work_items,
            max_splits=max_splits,
            kv_tile_size=kv_tile_size,
        )
        return sanitized, schedule
    if seq_lens.ndim != 1:
        raise ValueError(
            f"seq_lens must be 1-D, got shape={tuple(seq_lens.shape)}"
        )
    if seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got dtype={seq_lens.dtype}")
    if target_work_items <= 0:
        raise ValueError(
            f"target_work_items must be positive, got {target_work_items}"
        )
    if base_work_items is None:
        base_work_items = target_work_items
    if not 0 < base_work_items <= target_work_items:
        raise ValueError(
            "base_work_items must be in [1, target_work_items], got "
            f"{base_work_items} for target_work_items={target_work_items}"
        )
    if max_splits <= 0:
        raise ValueError(f"max_splits must be positive, got {max_splits}")
    if kv_tile_size <= 0:
        raise ValueError(f"kv_tile_size must be positive, got {kv_tile_size}")

    batch_size = seq_lens.numel()
    if target_work_items < batch_size:
        raise ValueError(
            "target_work_items must provide at least one slot per request, got "
            f"{target_work_items} for batch_size={batch_size}"
        )
    if base_work_items < batch_size:
        raise ValueError(
            "base_work_items must be at least batch_size because every request "
            f"requires one split, got {base_work_items} for batch_size={batch_size}"
        )
    block_size = triton.next_power_of_2(batch_size)
    if block_size > 1024:
        raise ValueError(
            f"ragged scheduler supports at most 1024 requests, got {batch_size}"
        )
    work_block_size = triton.next_power_of_2(target_work_items)
    if work_block_size > 65536:
        raise ValueError(
            "ragged scheduler supports at most 65536 work items, "
            f"got {target_work_items}"
        )

    split_counts = torch.empty_like(seq_lens)
    split_offsets = torch.empty(
        batch_size + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    worklist = torch.empty(
        (target_work_items, 2),
        dtype=torch.int32,
        device=seq_lens.device,
    )
    sanitized_block_tables = torch.empty_like(block_tables)
    copy_block_size = 256
    block_table_elements = block_tables.numel()
    copy_programs = triton.cdiv(block_table_elements, copy_block_size)
    _plan_ragged_splits_kernel[(copy_programs,)](
        seq_lens,
        split_counts,
        split_offsets,
        worklist,
        block_tables,
        sanitized_block_tables,
        block_table_elements,
        batch_size,
        base_work_items,
        target_work_items,
        max_splits,
        KV_TILE_SIZE=kv_tile_size,
        BLOCK_SIZE=block_size,
        BATCH_SIZE=batch_size,
        WORK_BLOCK_SIZE=work_block_size,
        COPY_BLOCK_SIZE=copy_block_size,
        SANITIZE_BLOCK_TABLES=True,
    )
    return sanitized_block_tables, (
        split_counts,
        split_offsets,
        worklist,
    )


def plan_ragged_splits(
    seq_lens: torch.Tensor,
    *,
    target_work_items: int,
    base_work_items: int | None = None,
    max_splits: int,
    kv_tile_size: int = 128,
) -> torch.Tensor:
    """Plan approximately equal-sized KV chunks without a host synchronization.

    This follows FlashMLA's scheduler policy: derive a target KV-block payload
    from total work, then split each request into enough chunks to approach that
    payload. The fixed ``max_splits`` extent keeps the launch CUDA-graph stable.
    """
    split_counts, _, _ = plan_ragged_schedule(
        seq_lens,
        target_work_items=target_work_items,
        base_work_items=base_work_items,
        max_splits=max_splits,
        kv_tile_size=kv_tile_size,
    )
    return split_counts
