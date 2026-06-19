"""Tests de las features forensics."""
import numpy as np

from src.detection.forensic import FEATURE_NAMES, extract_features
from src.generation.classical import generate_fake, make_base_image


def test_feature_vector_shape_and_finite():
    base = make_base_image(np.random.default_rng(0), size=64)
    feats = extract_features(base)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()


def test_fingerprint_changes_features():
    rng = np.random.default_rng(0)
    base = make_base_image(rng, size=64)
    fake = generate_fake(base, "sd-inpaint", np.random.default_rng(1))
    assert not np.array_equal(extract_features(base), extract_features(fake))
