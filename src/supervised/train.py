from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def build_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=0.01, max_iter=2000, class_weight="balanced", random_state=42)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6,
            min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
    }
