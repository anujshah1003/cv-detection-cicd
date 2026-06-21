"""
Tests for the object detector.

These fall into two categories, deliberately separated:

1. CODE-LEVEL tests: normal software testing. Do we raise the right
   errors? Do data classes behave correctly? These would look the same
   in any Python project.

2. MODEL-LEVEL (sanity / behavioral) tests: ML-specific. Does the model
   actually load? Does inference run end-to-end without crashing? Are
   output shapes and types correct? These catch a different class of bug
   that normal unit tests miss entirely -- e.g. a code change that's
   syntactically fine but silently breaks preprocessing, or an
   incompatible model file.

In a real production project you would also add:
  - Regression tests against a fixed "golden" image + expected detections,
    to catch accuracy regressions when the model or preprocessing changes.
  - Latency/performance tests (inference under N ms).
We keep those conceptual here and explain why, since they need a curated
real-world test image and a baseline run to compare against.
"""
from pathlib import Path

import pytest

from app.detector import ObjectDetector, Detection

TEST_IMAGE = str(Path(__file__).parent / "assets" / "test_image.jpg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    """
    Load the model once per test module, not once per test.
    Loading YOLO weights is slow (and in CI, costs time = money), so we
    reuse a single loaded model across all tests in this file.
    """
    return ObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.25)


# ---------------------------------------------------------------------------
# Code-level tests
# ---------------------------------------------------------------------------

def test_detection_dataclass_holds_expected_fields():
    d = Detection(class_name="person", confidence=0.9, box=[0.0, 0.0, 10.0, 10.0])
    assert d.class_name == "person"
    assert d.confidence == 0.9
    assert len(d.box) == 4


def test_predict_raises_on_missing_file(detector):
    with pytest.raises(FileNotFoundError):
        detector.predict("tests/assets/does_not_exist.jpg")


# ---------------------------------------------------------------------------
# Model-level (sanity) tests
# ---------------------------------------------------------------------------

def test_model_loads_successfully():
    """
    The single most important ML CI check: does the model even load?
    This catches broken weight files, version mismatches, and missing
    dependencies before they reach production.
    """
    detector = ObjectDetector(model_path="yolov8n.pt")
    assert detector.model is not None


def test_inference_runs_end_to_end(detector):
    """
    Smoke test: run inference on a known image and confirm the pipeline
    completes without error and returns a well-formed result -- this is
    NOT asserting the model found a specific object (our checked-in test
    image is synthetic), just that the full preprocessing -> forward pass
    -> postprocessing pipeline executes correctly.
    """
    detections = detector.predict(TEST_IMAGE)
    assert isinstance(detections, list)


def test_detection_output_schema(detector):
    """
    If the model detects anything, validate the SHAPE and TYPES of the
    output. This is what catches silent breakage -- e.g. someone changes
    postprocessing and confidence becomes a numpy float32 instead of a
    Python float, which would break JSON serialization in the API layer
    without raising any error in casual testing.
    """
    detections = detector.predict(TEST_IMAGE)
    for d in detections:
        assert isinstance(d.class_name, str)
        assert isinstance(d.confidence, float)
        assert 0.0 <= d.confidence <= 1.0
        assert isinstance(d.box, list)
        assert len(d.box) == 4
        assert all(isinstance(c, float) for c in d.box)


def test_confidence_threshold_is_respected(detector):
    """All returned detections must be at or above the configured threshold."""
    detections = detector.predict(TEST_IMAGE)
    for d in detections:
        assert d.confidence >= detector.confidence_threshold
