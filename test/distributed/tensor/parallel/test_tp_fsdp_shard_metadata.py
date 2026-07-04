# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestGetBoxForUnevenShards(DTensorTestBase):
    """Regression test for _get_box_for with uneven shard sizes.

    _get_box_for must compute correct per-rank offsets and sizes for
    DTensors where the shard dimension is not evenly divisible by the
    mesh size. The old implementation used floor division which produced
    wrong offsets for non-zero ranks.
    """

    @property
    def backend(self):
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 3

    def _get_fsdp_helpers(self):
        """Import _get_box_for and _get_local_box via sys.modules to avoid circular import."""
        import torch.distributed.fsdp  # noqa: F401 — trigger full import

        mod = sys.modules["torch.distributed.tensor.parallel.fsdp"]
        return mod._get_box_for, mod._get_local_box

    @with_comms
    def test_get_box_for_uneven_shard_offsets(self):
        """Test offsets are correct when global size is not divisible by world size.

        global_size=5, world_size=3: torch.chunk produces [2, 2, 1].
        """
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        dt = distribute_tensor(
            torch.arange(5, dtype=torch.float32), mesh, [Shard(0)]
        )
        # torch.chunk(5, 3) → [2, 2, 1], offsets = [0, 2, 4]
        expected_offsets = [0, 2, 4]
        expected_sizes = [2, 2, 1]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_uneven_ceil_division(self):
        """Test that ceil-division (torch.chunk) semantics are used, not floor-division.

        global_size=7, world_size=3: torch.chunk produces [3, 3, 1] (ceil),
        NOT [3, 2, 2] (floor / tensor_split).
        """
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        dt = distribute_tensor(
            torch.arange(7, dtype=torch.float32), mesh, [Shard(0)]
        )
        # torch.chunk(7, 3) → [3, 3, 1], offsets = [0, 3, 6]
        expected_offsets = [0, 3, 6]
        expected_sizes = [3, 3, 1]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_uneven_larger(self):
        """Test with global_size=10, world_size=3 where chunk/tensor_split diverge.

        torch.chunk(10, 3) → [4, 4, 2], offsets = [0, 4, 8]
        """
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        dt = distribute_tensor(
            torch.arange(10, dtype=torch.float32), mesh, [Shard(0)]
        )
        expected_offsets = [0, 4, 8]
        expected_sizes = [4, 4, 2]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_even_shards_regression(self):
        """Ensure even sharding still works correctly after the fix."""
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        # global_size=6, world_size=3 → [2, 2, 2] (even)
        dt = distribute_tensor(
            torch.arange(6, dtype=torch.float32), mesh, [Shard(0)]
        )
        expected_offsets = [0, 2, 4]
        expected_sizes = [2, 2, 2]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_global_size_less_than_world(self):
        """Edge case: global_size < world_size → some ranks get empty shards."""
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        # global_size=1, world_size=3 → torch.chunk(1, 3) → [1] with 2 empty
        dt = distribute_tensor(
            torch.arange(1, dtype=torch.float32), mesh, [Shard(0)]
        )
        # torch.chunk(1, 3) → chunk_size = ceil(1/3) = 1
        # rank 0: offset=0, size=1; rank 1: offset=1, size=0; rank 2: offset=1, size=0
        expected_offsets = [0, 1, 1]
        expected_sizes = [1, 0, 0]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_replicated_placement(self):
        """Non-shard placement: offsets should be zero, sizes should be full global size."""
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        dt = distribute_tensor(
            torch.arange(5, dtype=torch.float32), mesh, [Replicate()]
        )

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], 0)
            self.assertEqual(sizes[0], 5)

    @with_comms
    def test_get_box_for_multidim_non_zero_shard_dim(self):
        """Test with a 2D tensor sharded on dim=1."""
        _get_box_for, _ = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        # Shape (4, 7), sharded on dim=1 → torch.chunk(7, 3) → [3, 3, 1]
        t = torch.randn(4, 7)
        dt = distribute_tensor(t, mesh, [Shard(1)])

        expected_offsets_dim1 = [0, 3, 6]
        expected_sizes_dim1 = [3, 3, 1]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            # dim 0 is not sharded
            self.assertEqual(offsets[0], 0)
            self.assertEqual(sizes[0], 4)
            # dim 1 is sharded
            self.assertEqual(offsets[1], expected_offsets_dim1[idx])
            self.assertEqual(sizes[1], expected_sizes_dim1[idx])

    @with_comms
    def test_get_local_box_matches_get_box_for(self):
        """Ensure _get_local_box returns the same result as _get_box_for for current rank."""
        _get_box_for, _get_local_box = self._get_fsdp_helpers()
        mesh = init_device_mesh("cpu", (self.world_size,))

        dt = distribute_tensor(
            torch.arange(7, dtype=torch.float32), mesh, [Shard(0)]
        )

        local_offsets, local_sizes = _get_local_box(dt)
        expected_offsets, expected_sizes = _get_box_for(dt, self.rank)

        self.assertEqual(local_offsets, expected_offsets)
        self.assertEqual(local_sizes, expected_sizes)


if __name__ == "__main__":
    run_tests()
