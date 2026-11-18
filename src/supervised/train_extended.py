"""Fixed expanded search; repeated source-stratified CV; freeze selection before Test."""
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from src.common.paths import DATA, MODEL, REPORT
from src.supervised.dataset import validate_manifest, sha256
from src.supervised.deep_features import VERSION, global_view
from src.supervised.improve import cached_features, choose_threshold
from src.supervised.run_supervised import measure, positive_scores, bootstrap_auc


def candidates():
    models = {}
    for view in ('global', 'global_lungs'):
        for c in (.0001, .001, .01, .1, 1., 10.):
            steps = [FunctionTransformer(global_view)] if view == 'global' else []
            models[f'extended_{view}_LR_C{c}'] = make_pipeline(*steps, StandardScaler(),
                LogisticRegression(C=c, class_weight='balanced', max_iter=5000, random_state=42))
        for c in (.01, .1, 1., 10.):
            steps = [FunctionTransformer(global_view)] if view == 'global' else []
            models[f'extended_{view}_RBF_C{c}'] = make_pipeline(*steps, StandardScaler(),
                SVC(C=c, class_weight='balanced', kernel='rbf', probability=True, random_state=42))
    return models


def per_source(frame, scores, threshold):
    return {source: measure(frame.label.to_numpy()[frame.source == source], scores[frame.source == source], threshold)
            for source in sorted(frame.source.unique())}


def main():
    frame = pd.read_csv(DATA / 'manifest_extended.csv')
    validate_manifest(frame, DATA)
    incumbent = joblib.load(MODEL)
    if incumbent['report'].get('experiment') == 'extended_v3':
        incumbent = joblib.load(MODEL.with_name('before_extended.joblib'))
    else:
        joblib.dump(incumbent, MODEL.with_name('before_extended.joblib'), compress=3)
    models = candidates()
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha256(DATA / 'manifest_extended.csv'),
            'candidates': list(models), 'folds': 5, 'repeats': 3, 'seed': 42,
            'selection': 'Highest mean of per-source ROC AUC over 15 Train folds; strata source + label.',
            'promotion': 'Validation macro source AUC and balanced accuracy must both exceed incumbent, with no JSRT validation AUC regression.',
            'threshold': 'Validation balanced accuracy only',
            'test': 'Frozen original JSRT Test plus new LIDC Test; evaluated after promotion decision. Small exploratory cohorts, not clinical validation.'}
    plan_path = REPORT.parent / 'experiment_v3.json'
    if plan_path.exists() and json.loads(plan_path.read_text())['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Dataset changed after experiment registration')
    plan_path.write_text(json.dumps(plan, indent=2))
    features, weights_hash = cached_features(frame)
    train = frame.split.eq('Train').to_numpy(); val = frame.split.eq('Validation').to_numpy(); test = frame.split.eq('Test').to_numpy()
    x, y = features[train], frame.label.to_numpy()[train]
    sources = frame.source.to_numpy()[train]
    strata = sources + '_' + y.astype(str)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(x, strata))
    cv = {}; started = time.perf_counter()
    for name, model in models.items():
        results = []
        for fit_ids, holdout_ids in folds:
            fitted = clone(model).fit(x[fit_ids], y[fit_ids])
            scores = positive_scores(fitted, x[holdout_ids])
            results.append(float(np.mean([roc_auc_score(y[holdout_ids][sources[holdout_ids] == s], scores[sources[holdout_ids] == s]) for s in np.unique(sources)])))
        cv[name] = {'auc_mean': float(np.mean(results)), 'auc_std': float(np.std(results)), 'folds': results}
        print(f'{name}: source-macro CV AUC {np.mean(results):.3f}', flush=True)
    best = max(cv, key=lambda k: cv[k]['auc_mean'])
    model = models[best].fit(x, y)
    val_frame = frame[val].reset_index(drop=True)
    val_scores = positive_scores(model, features[val])
    threshold = choose_threshold(val_frame.label.to_numpy(), val_scores)
    validation = measure(val_frame.label, val_scores, threshold)
    validation['train_seconds'] = round(time.perf_counter() - started, 3)
    by_source = per_source(val_frame, val_scores, threshold)
    old_val_scores = positive_scores(incumbent['model'], features[val])
    old_val = per_source(val_frame, old_val_scores, incumbent['report']['threshold'])
    approved = all(np.mean([v[key] for v in by_source.values()]) > np.mean([v[key] for v in old_val.values()]) for key in ('roc_auc', 'balanced_accuracy')) and by_source['JSRT']['roc_auc'] >= old_val['JSRT']['roc_auc']
    selection = {**plan, 'selected_model': best, 'threshold': threshold, 'cross_validation': cv,
                 'validation_by_source': by_source, 'incumbent_validation_by_source': old_val, 'promoted': bool(approved)}
    (REPORT.parent / 'selection_v3.json').write_text(json.dumps(selection, indent=2))
    # No test scores are used above this line.
    test_frame = frame[test].reset_index(drop=True)
    scores = positive_scores(model, features[test])
    test_metrics = measure(test_frame.label, scores, threshold)
    test_metrics['roc_auc_ci95_bootstrap'] = bootstrap_auc(test_frame.label.to_numpy(), scores)
    test_sources = per_source(test_frame, scores, threshold)
    old_test = per_source(test_frame, positive_scores(incumbent['model'], features[test]), incumbent['report']['threshold'])
    quality = json.loads((DATA / 'quality_extended.json').read_text())
    report = {**incumbent['report'], 'experiment': 'extended_v3', 'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset': 'JSRT + public LIDC-IDRI frontal X-rays', 'dataset_size': len(frame),
              'manifest_filename': 'manifest_extended.csv', 'manifest_sha256': plan['manifest_sha256'],
              'split_counts': frame.split.value_counts().to_dict(), 'state_counts': frame.state.value_counts().to_dict(),
              'source_counts': frame.source.value_counts().to_dict(), 'selected_model': best, 'threshold': threshold,
              'selection_rule': plan['selection'], 'cross_validation': cv, 'cv_repeats': 3,
              'validation': {best: validation}, 'validation_by_source': by_source,
              'test': test_metrics, 'test_by_source': test_sources, 'incumbent_test_by_source': old_test,
              'test_fixed_threshold_05': measure(test_frame.label, scores, .5),
              'test_majority_baseline_accuracy': float((test_frame.label == 0).mean()),
              'promoted': bool(approved), 'data_quality': quality, 'baseline': None,
              'test_status': plan['test'],
              'limitations': [quality['limitations'],
                 'Les annotations LIDC sont au niveau du patient ; celles de JSRT décrivent un nodule. Leur mélange reste exploratoire.',
                 'Très peu de nouveaux cas malins et test externe minuscule : aucune précision clinique démontrée.',
                 'Le test JSRT a déjà été observé ; le nouveau test LIDC ne doit plus servir au réglage.',
                 'Le score n’est pas une probabilité individuelle de cancer.',
                 'Préentraînement thoracique public ; recoupement inter-bases non audité au niveau patient.',
                 'Les caractéristiques DenseNet restent gelées. Recherche étendue des classifieurs, sans fine-tuning du réseau.',
                 'Une classe négative ne prouve pas l’absence de cancer.']}
    report.pop('baseline')
    bundle = {**incumbent, 'model': model, 'report': report, 'encoder_sha256': weights_hash,
              'dataset_hashes': dict(zip(frame.sha256, frame.split))}
    joblib.dump(bundle, MODEL.with_name('candidate_v3.joblib'), compress=3)
    (REPORT.parent / 'candidate_v3.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    predictions = test_frame[['patient_id', 'source', 'state', 'label']].copy()
    predictions['score'] = scores; predictions['prediction'] = (scores >= threshold).astype(int)
    predictions.to_csv(REPORT.parent / 'test_predictions_v3.csv', index=False)
    frame.to_csv(REPORT.parent / 'split_manifest_v3.csv', index=False)
    (REPORT.parent / 'data_quality_v3.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    if approved:
        temporary = MODEL.with_suffix('.tmp'); joblib.dump(bundle, temporary, compress=3); temporary.replace(MODEL)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        predictions.to_csv(REPORT.parent / 'test_predictions.csv', index=False)
        frame.to_csv(REPORT.parent / 'split_manifest.csv', index=False)
    print(json.dumps({'promoted': bool(approved), 'best': best, 'validation': by_source, 'test': test_sources}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
