from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def build_models():
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )),
        ]),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42
            )),
        ]),

        "random_forest": Pipeline([
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }

    return models