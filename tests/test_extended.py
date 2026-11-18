"""Guard against incorrect diagnoses, lateral views and patient leakage."""
import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from src.common.paths import DATA
from src.supervised.extend_dataset import confirmed_label, frontal_evidence, to_grayscale


def test_diagnosis_is_not_inferred_from_nodule_or_unknown():
    assert confirmed_label(2, 2) == 1
    assert confirmed_label(2, 3) == 1
    assert confirmed_label(1, 1) == 0
    assert confirmed_label(1, 3) == 0
    for diagnosis, method in [(0, 0), (2, 0), (2, 4), (3, 2), (1, 4)]:
        assert confirmed_label(diagnosis, method) is None


def test_frontal_view_requires_documented_orientation():
    ds = Dataset(); ds.Modality = 'DX'
    assert frontal_evidence(ds) is None
    ds.PatientOrientation = ['P', 'F']
    assert frontal_evidence(ds) is None
    ds.PatientOrientation = ['L', 'F']
    assert frontal_evidence(ds)
    ds.ViewPosition = 'LL'
    assert frontal_evidence(ds) is None  # Explicit lateral beats fallback orientation.
    ds.ViewPosition = 'PA'
    assert frontal_evidence(ds)
    ds.Modality = 'CT'
    assert frontal_evidence(ds) is None


def test_dicom_polarity():
    ds = Dataset(); ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows = ds.Columns = 16; ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    ds.PixelData = np.arange(256, dtype=np.uint16).reshape(16, 16).tobytes()
    ds.PhotometricInterpretation = 'MONOCHROME2'
    normal = np.asarray(to_grayscale(ds))
    ds.PhotometricInterpretation = 'MONOCHROME1'
    inverse = np.asarray(to_grayscale(ds))
    assert normal[0, 0] == 0 and normal[-1, -1] == 255
    np.testing.assert_allclose(normal.astype(int) + inverse.astype(int), 255, atol=1)


@pytest.mark.skipif(not (DATA / 'manifest_extended.csv').exists(), reason='New data not downloaded')
def test_extended_patients_and_original_holdouts():
    from src.supervised.dataset import validate_manifest
    old = pd.read_csv(DATA / 'manifest_2048.csv').set_index('patient_id')
    new = pd.read_csv(DATA / 'manifest_extended.csv').set_index('patient_id')
    validate_manifest(new.reset_index(), DATA)
    pd.testing.assert_frame_equal(old[['split', 'label']], new.loc[old.index, ['split', 'label']])
    extra = new[~new.index.isin(old.index)]
    assert len(extra) > 0
    assert set(extra.source) == {'LIDC-IDRI'}
    assert extra.view_evidence.notna().all()
    assert all(confirmed_label(r.diagnosis, r.diagnosis_method) == r.label for r in extra.itertuples())
    assert set(extra.split) == {'Train', 'Validation', 'Test'}


@pytest.mark.skipif(not (DATA / 'manifest_extended.csv').exists(), reason='New data not downloaded')
def test_saved_512_candidate_matches_evaluation():
    from src.common.paths import ROOT, MODEL
    from src.supervised.inference import load_model, predict
    candidate = MODEL.with_name('candidate_v4.joblib')
    if not candidate.exists():
        pytest.skip('512 candidate not trained')
    model = load_model(candidate)
    manifest = pd.read_csv(DATA / 'manifest_extended.csv')
    predictions = pd.read_csv(ROOT / 'reports/supervised/test_predictions_v4.csv').set_index('patient_id')
    for source in ('JSRT', 'LIDC-IDRI'):
        row = manifest[(manifest.source == source) & (manifest.split == 'Test')].iloc[0]
        result = predict((DATA / row.path).read_bytes(), model)
        assert result['score'] == pytest.approx(predictions.loc[row.patient_id, 'score'], abs=1e-6)
        assert result['known_dataset_split'] == 'Test'
        assert result['research_only']
