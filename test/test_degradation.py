from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.degradation import (
    apply_degradation,
    apply_exposure_variation,
    apply_motion_blur,
    build_motion_blur_kernel,
    derive_degradation_seed,
    sample_degradation_params,
    sample_degradation_type,
)


def _make_gradient_image(height=17, width=19):
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    image = np.stack(
        [
            np.broadcast_to(cols, (height, width)),
            np.broadcast_to(rows, (height, width)),
            (np.broadcast_to(cols, (height, width)) + np.broadcast_to(rows, (height, width))) / 2.0,
        ],
        axis=-1,
    )
    return np.clip(image, 0, 255).astype(np.uint8)


class TestDegradationCore(unittest.TestCase):
    def test_motion_blur_kernel_is_normalized_and_validates_size(self):
        kernel = build_motion_blur_kernel(kernel_size=15, angle_deg=37.5)

        self.assertEqual(kernel.shape, (15, 15))
        self.assertEqual(kernel.dtype, np.float32)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)
        self.assertGreater(np.count_nonzero(kernel), 0)

        with self.assertRaises(ValueError):
            build_motion_blur_kernel(kernel_size=8, angle_deg=0.0)

    def test_degradation_preserves_shape_and_dtype(self):
        image = _make_gradient_image()

        blurred = apply_motion_blur(image, kernel_size=7, angle_deg=25.0)
        exposed = apply_exposure_variation(image, gain=1.7, gamma=0.75)

        self.assertEqual(blurred.shape, image.shape)
        self.assertEqual(exposed.shape, image.shape)
        self.assertEqual(blurred.dtype, image.dtype)
        self.assertEqual(exposed.dtype, image.dtype)
        self.assertFalse(np.array_equal(blurred, image))
        self.assertFalse(np.array_equal(exposed, image))

    def test_sampling_is_reproducible_and_seed_uses_all_key_fields(self):
        params_a = sample_degradation_params("exposure", "medium", seed=123)
        params_b = sample_degradation_params("exposure", "medium", seed=123)
        params_c = sample_degradation_params("exposure", "medium", seed=124)

        self.assertEqual(params_a, params_b)
        self.assertNotEqual(params_a, params_c)

        base = derive_degradation_seed(42, "MH_01_easy", 7, 0, "motion_blur")
        self.assertEqual(base, derive_degradation_seed(42, "MH_01_easy", 7, 0, "motion_blur"))
        self.assertNotEqual(base, derive_degradation_seed(42, "MH_02_easy", 7, 0, "motion_blur"))
        self.assertNotEqual(base, derive_degradation_seed(42, "MH_01_easy", 7, 1, "motion_blur"))
        self.assertNotEqual(base, derive_degradation_seed(42, "MH_01_easy", 7, 0, "exposure"))

    def test_apply_degradation_clean_and_mixed_metadata(self):
        image = _make_gradient_image()

        clean_image, clean_metadata = apply_degradation(
            image,
            {"type": "clean", "severity": "none", "seed": None},
        )
        self.assertTrue(np.array_equal(clean_image, image))
        self.assertEqual(clean_metadata["degradation_type"], "clean")
        self.assertEqual(clean_metadata["severity"], "none")
        self.assertIsNone(clean_metadata["seed"])

        mixed_config = sample_degradation_params("mixed", "medium", seed=321)
        mixed_image, mixed_metadata = apply_degradation(image, mixed_config)

        self.assertEqual(mixed_image.shape, image.shape)
        self.assertEqual(mixed_image.dtype, image.dtype)
        self.assertFalse(np.array_equal(mixed_image, image))
        self.assertEqual(mixed_metadata["degradation_type"], "mixed")
        self.assertIn("motion_blur", mixed_metadata["params"])
        self.assertIn("exposure", mixed_metadata["params"])
        self.assertEqual(mixed_metadata["params"]["order"], ["motion_blur", "exposure"])

    def test_weighted_type_sampling_is_deterministic(self):
        weights = {"clean": 0.0, "motion_blur": 0.0, "exposure": 1.0, "mixed": 0.0}

        self.assertEqual(sample_degradation_type(weights, seed=5), "exposure")
        self.assertEqual(sample_degradation_type(weights, seed=5), "exposure")

        with self.assertRaises(ValueError):
            sample_degradation_type({"clean": 0.0}, seed=5)


if __name__ == "__main__":
    unittest.main()
