"""Public TCIA LIDC-IDRI frontal X-rays with conservative patient-level labels."""
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import requests
from PIL import Image

from src.common.paths import DATA, ROOT
from src.supervised.dataset import sha256, split_manifest, validate_manifest

API = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/'
DIAGNOSES = 'https://www.cancerimagingarchive.net/wp-content/uploads/tcia-diagnosis-data-2012-04-20.xls'
EXTERNAL = ROOT / 'data/external/lidc/downloads'


def confirmed_label(diagnosis, method):
    # Exclude unknowns, metastases and progression-only diagnoses.
    if diagnosis == 2 and method in (2, 3):
        return 1
    if diagnosis == 1 and method in (1, 2, 3):
        return 0
    return None


def frontal_evidence(ds):
    if ds.Modality not in ('DX', 'CR'):
        return None
    view = str(getattr(ds, 'ViewPosition', '')).strip().upper()
    if view:
        return 'ViewPosition:' + view if view in ('PA', 'AP') else None
    # Row direction left/right, column direction feet identifies frontal projection.
    orientation = tuple(getattr(ds, 'PatientOrientation', []))
    if orientation in (('L', 'F'), ('R', 'F')):
        return 'PatientOrientation:' + '/'.join(orientation)
    return None


def to_grayscale(ds):
    if ds.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('Non-monochrome radiograph')
    a = ds.pixel_array.astype(np.float32)
    if a.ndim != 2:
        raise ValueError('Expected a single 2D radiograph')
    a = a * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
    # Some TCIA DX files contain CT-like window values (-600/1600).
    # Use the same robust image-only intensity normalization for every new case.
    lo, hi = np.percentile(a, [.5, 99.5])
    if hi <= lo:
        raise ValueError('No image contrast')
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        a = 1 - a
    image = Image.fromarray(np.round(a * 255).astype('uint8'))
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image


def fetch_series(series):
    patient = series['PatientID']
    path = EXTERNAL / (patient + '.zip')
    if not path.exists():
        response = requests.get(API + 'getImage', params={'SeriesInstanceUID': series['SeriesInstanceUID']}, timeout=(20, 180))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if archive.testzip() is not None:
                raise ValueError('Corrupted archive')
        temporary = path.with_suffix('.part')
        temporary.write_bytes(response.content)
        temporary.replace(path)
    candidates = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith('.dcm'):
                continue
            ds = pydicom.dcmread(io.BytesIO(archive.read(name)))
            evidence = frontal_evidence(ds)
            if evidence and str(ds.PatientID) == patient and 'ORIGINAL' in getattr(ds, 'ImageType', []):
                candidates.append((str(ds.SOPInstanceUID), ds, evidence))
    if not candidates:
        print(patient + ': no documented frontal view', flush=True)
        return {'patient_id': patient, 'excluded': 'No documented original frontal projection'}
    _, ds, evidence = sorted(candidates, key=lambda item: item[0])[0]
    output = DATA / 'images_lidc' / (patient + '.png')
    to_grayscale(ds).save(output)
    print(patient + ': frontal radiograph added', flush=True)
    return {'patient_id': patient, 'path': str(output.relative_to(DATA)), 'sha256': sha256(output),
            'dicom_sha256': sha256(path), 'series_uid': series['SeriesInstanceUID'],
            'sop_uid': str(ds.SOPInstanceUID), 'view_evidence': evidence, 'source': 'LIDC-IDRI',
            'license': series['LicenseURI']}


def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    (DATA / 'images_lidc').mkdir(exist_ok=True)
    if not (EXTERNAL / 'diagnoses.xls').exists():
        response = requests.get(DIAGNOSES, timeout=60); response.raise_for_status()
        (EXTERNAL / 'diagnoses.xls').write_bytes(response.content)
    if sha256(EXTERNAL / 'diagnoses.xls') != '5b7cdcb0fc016e976f5c65d8122bb93c3665432d44e1573bac1cc9573079868a':
        raise ValueError('Diagnosis reference file changed; review it before proceeding')
    listing = EXTERNAL / 'series.json'
    if not listing.exists():
        response = requests.get(API + 'getSeries', params={'Collection': 'LIDC-IDRI', 'Modality': 'DX'}, timeout=60)
        response.raise_for_status(); listing.write_text(json.dumps(response.json(), indent=2))
    diagnoses = pd.read_excel(EXTERNAL / 'diagnoses.xls').iloc[:, :3]
    diagnoses.columns = ['patient_id', 'diagnosis', 'diagnosis_method']
    diagnoses['label'] = [confirmed_label(d, m) for d, m in zip(diagnoses.diagnosis, diagnoses.diagnosis_method)]
    eligible = diagnoses.dropna(subset=['label']).set_index('patient_id')
    series = [s for s in json.loads(listing.read_text()) if s['PatientID'] in eligible.index]
    if len({s['PatientID'] for s in series}) != len(series):
        raise ValueError('Multiple series per patient require explicit resolution')
    print(f'{len(series)} eligible new patients', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch_series, series))
    included = pd.DataFrame([r for r in results if 'excluded' not in r]).merge(eligible.reset_index(), on='patient_id')
    included['label'] = included.label.astype(int)
    included['state'] = included.label.map({0: 'benign', 1: 'malignant'})
    included = split_manifest(included)
    jsrt = pd.read_csv(DATA / 'manifest_2048.csv').assign(source='JSRT')
    merged = pd.concat([jsrt, included], ignore_index=True)
    validate_manifest(merged, DATA)
    merged.to_csv(DATA / 'manifest_extended.csv', index=False)
    audit = {'source': 'https://www.cancerimagingarchive.net/collection/lidc-idri/',
             'diagnoses_url': DIAGNOSES, 'diagnoses_sha256': sha256(EXTERNAL / 'diagnoses.xls'),
             'series_sha256': sha256(listing), 'new_patients': len(included), 'total_patients': len(merged),
             'new_split_counts': {split: subset.state.value_counts().to_dict() for split, subset in included.groupby('split')},
             'exclusions': [r for r in results if 'excluded' in r],
             'label_policy': 'Patient-level primary lung malignancy confirmed by biopsy/resection; benign confirmed by stability >=2 years, biopsy or resection. Unknown, metastases and progression-only excluded.',
             'limitations': 'Diagnosis is patient-level and may concern a lesion invisible on the chest X-ray. Temporal correspondence with diagnosis is not established. Cross-source identity linkage is unavailable.',
             'normalization': 'DICOM rescale, 0.5/99.5 image percentiles, MONOCHROME1 inversion; no supplied VOI window; PNG 8bit max2048px; original images only, derived images excluded',
             'license': 'CC BY 3.0; attribution: Armato SG et al. LIDC-IDRI, TCIA, DOI 10.7937/K9/TCIA.2015.LO9QL9SX'}
    (DATA / 'quality_extended.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    return merged


if __name__ == '__main__':
    main()
