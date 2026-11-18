import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.common.paths import DATA, MODEL, ROOT
from src.supervised.dataset import STATES, split_manifest, validate_manifest
from src.supervised.features import extract_features
from src.supervised.inference import load_model, predict
from src.supervised.preprocess import MAX_BYTES, read_image
from src.supervised.run_supervised import measure


def encoded(array, fmt="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


def test_invalid_files():
    for content in [b"", b"not an image", b"x" * (MAX_BYTES + 1)]:
        with pytest.raises(ValueError):
            read_image(content)


@pytest.mark.parametrize("array", [
    np.zeros((256, 256), dtype=np.uint8),
    np.zeros((32, 32), dtype=np.uint8),
    np.zeros((128, 1000), dtype=np.uint8),
    np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8),
    np.full((256, 256), 4095, dtype=np.uint16),
    np.zeros((256, 256, 4), dtype=np.uint8),
])
def test_reject_incompatible_images(array):
    with pytest.raises(ValueError):
        read_image(encoded(array))


def test_reject_wrong_format():
    with pytest.raises(ValueError, match="Formats"):
        read_image(encoded(np.zeros((256, 256), dtype=np.uint8), "BMP"))


def test_training_and_inference_features_match(tmp_path):
    array = np.random.default_rng(42).integers(0, 256, (256, 256), dtype=np.uint8)
    content = encoded(array)
    path = tmp_path / "image.png"
    path.write_bytes(content)
    np.testing.assert_array_equal(extract_features(path), extract_features(read_image(content)))


def test_malignancy_not_nodule_presence():
    assert STATES == {"malignant": 1, "benign": 0, "non-nodule": 0}


def test_reproducible_patient_splits():
    frame = pd.DataFrame({"patient_id": range(247), "state": ["malignant"] * 100 + ["benign"] * 54 + ["non-nodule"] * 93})
    first = split_manifest(frame)
    pd.testing.assert_frame_equal(first, split_manifest(frame))
    assert first.split.value_counts().to_dict() == {"Train": 147, "Validation": 50, "Test": 50}
    assert first.patient_id.nunique() == 247
    for _, subset in first.groupby("split"):
        assert set(subset.state) == set(STATES)


def test_duplicate_manifest_rejected(tmp_path):
    frame = pd.DataFrame([dict(path="a.jpg", patient_id="p", state="benign", label=0, split="Train", sha256="same")] * 2)
    with pytest.raises(ValueError, match="dupliqués"):
        validate_manifest(frame, tmp_path)


def test_metrics_confusion_order():
    metrics = measure(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.4]))
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["sensitivity"] == metrics["specificity"] == 0.5


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


@pytest.mark.skipif(not MODEL.exists(), reason="Run dataset preparation and training for integration tests")
def test_real_model_inference():
    bundle = load_model()
    frame = pd.read_csv(DATA / ("manifest_2048.csv" if bundle.get("schema_version") == 2 else "manifest.csv"))
    row = frame[frame.split == "Test"].iloc[0]
    content = (DATA / row.path).read_bytes()
    result = predict(content, bundle)
    assert result["research_only"] is True
    assert 0 <= result["score"] <= 1
    assert result["known_dataset_split"] == "Test"
    assert result["predicted_class"] == int(result["score"] >= result["threshold"])
    assert "ne confirme ni n'exclut" in result["interpretation"]
    assert predict(content, bundle)["score"] == pytest.approx(result["score"], abs=1e-12)


@pytest.mark.skipif(not MODEL.exists(), reason="Local model needed")
def test_app_example_end_to_end():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.button[0].disabled
    app.radio[0].set_value("Essayer un exemple du jeu de test").run()
    assert not app.exception
    app.checkbox[0].check().run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception
    assert any("Classe «" in element.value for element in app.info)
    assert any("Annotation du jeu de test" in element.value for element in app.markdown)
    app.selectbox[0].select_index(1).run()
    # An old result must not remain attached to a newly selected image.
    assert not any("Classe «" in element.value for element in app.info)


def test_threshold_selected_from_validation():
    from src.supervised.improve import choose_threshold
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .2, .3, .4])
    threshold = choose_threshold(labels, scores)
    assert .2 < threshold < .3
    assert measure(labels, scores, threshold)["balanced_accuracy"] == 1


@pytest.mark.skipif(not (DATA / "manifest_2048.csv").exists(), reason="High resolution data needed")
def test_resolution_upgrade_preserves_patient_splits():
    old = pd.read_csv(DATA / "manifest.csv").set_index("patient_id")
    new = pd.read_csv(DATA / "manifest_2048.csv").set_index("patient_id")
    pd.testing.assert_frame_equal(old[["split", "label", "state"]], new[["split", "label", "state"]])
    assert new.sha256.nunique() == len(new) == 246
    assert "JPCNN014" not in new.index
    for path in new.path:
        with Image.open(DATA / path) as image:
            assert image.size == (2048, 2048)
            assert image.mode == "L"


@pytest.mark.skipif(not (DATA / "manifest_2048.csv").exists(), reason="Deep model needed")
def test_deep_train_inference_parity():
    from src.supervised.deep_features import extract_deep_features
    bundle = load_model()
    if bundle.get("schema_version") != 2:
        pytest.skip("No deployed deep model")
    frame = pd.read_csv(DATA / "manifest_2048.csv")
    row = frame[frame.split == "Test"].iloc[0]
    path = DATA / row.path
    content = path.read_bytes()
    np.testing.assert_allclose(extract_deep_features(path), extract_deep_features(read_image(content)), atol=1e-5)
    predictions = pd.read_csv(ROOT / "reports/supervised/test_predictions.csv")
    expected = predictions[predictions.patient_id == row.patient_id].score.iloc[0]
    assert predict(content, bundle)["score"] == pytest.approx(expected, abs=1e-6)


def test_grayscale_palette_supported_and_color_palette_rejected():
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    gray = Image.fromarray(arr).convert("P")
    stream = io.BytesIO()
    gray.save(stream, format="PNG")
    assert read_image(stream.getvalue()).mode == "L"
    color = Image.new("P", (256, 256), 1)
    color.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    stream = io.BytesIO()
    color.save(stream, format="PNG")
    with pytest.raises(ValueError, match="couleur"):
        read_image(stream.getvalue())
