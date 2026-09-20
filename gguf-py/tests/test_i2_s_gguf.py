#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import quantize_i2_s  # noqa: E402


class TestI2SGGUF(unittest.TestCase):
    def test_ternary_packing_matches_i2_s_layout(self) -> None:
        source = np.concatenate((
            np.full(32, -1.0, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
        ))
        packed = quantize_i2_s(source)

        # Every byte packs one value from each 32-element group: -1, 0, +1,
        # 0 -> 00 01 10 01.  The trailing eight floats contain the scale.
        self.assertEqual(packed.shape, (64,))
        np.testing.assert_array_equal(packed[:32], np.full(32, 0x19, dtype=np.uint8))
        np.testing.assert_array_equal(packed[32:].view(np.float32), np.ones(8, dtype=np.float32))

    def test_tensor_global_scale_storage(self) -> None:
        logical_shape = (3, 128)
        packed = np.arange(np.prod(logical_shape) // 4 + 32, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "i2_s.gguf"
            writer = GGUFWriter(path, "clip")
            writer.add_tensor(
                "weight",
                packed,
                raw_shape=logical_shape,
                raw_dtype=GGMLQuantizationType.I2_S,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            tensor = GGUFReader(path).tensors[0]
            self.assertEqual(tensor.tensor_type, GGMLQuantizationType.I2_S)
            self.assertEqual(tensor.n_elements, np.prod(logical_shape))
            self.assertEqual(tensor.n_bytes, packed.size)
            np.testing.assert_array_equal(tensor.data, packed)

    def test_rejects_missing_scale_storage(self) -> None:
        writer = GGUFWriter("unused.gguf", "clip")
        with self.assertRaisesRegex(ValueError, "Invalid I2_S tensor"):
            writer.add_tensor(
                "weight",
                np.zeros(3 * 128 // 4, dtype=np.uint8),
                raw_shape=(3, 128),
                raw_dtype=GGMLQuantizationType.I2_S,
            )


if __name__ == "__main__":
    unittest.main()
