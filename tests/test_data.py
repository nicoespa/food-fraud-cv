"""Tests del corpus: integridad de splits (sin leakage) y held-out solo en test."""
import numpy as np

from src.data.corpus import CLASSES, build_corpus


def _cfg():
    return {
        "seed": 1,
        "data": {"image_size": 64, "holdout_generator": "instruct-edit",
                 "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "generation": {"generators": [{"name": "sd-inpaint"}, {"name": "img2img"},
                                       {"name": "instruct-edit"}]},
    }


def test_corpus_has_three_classes(tmp_path):
    df = build_corpus(_cfg(), tmp_path, n_sources=20)
    assert set(df["label"]) == set(CLASSES)


def test_no_source_leakage_across_splits(tmp_path):
    df = build_corpus(_cfg(), tmp_path, n_sources=20)
    # cada source_id cae en EXACTAMENTE un split
    per_source_splits = df.groupby("source_id")["split"].nunique()
    assert (per_source_splits == 1).all()


def test_holdout_generator_only_in_test(tmp_path):
    df = build_corpus(_cfg(), tmp_path, n_sources=20)
    held = df[df["generator"] == "instruct-edit"]
    assert len(held) > 0
    assert set(held["split"]) == {"test"}


def test_manifest_written(tmp_path):
    build_corpus(_cfg(), tmp_path, n_sources=12)
    assert (tmp_path / "manifest.csv").exists()
