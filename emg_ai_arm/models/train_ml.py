import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from emg_ai_arm.features.extract_features import extract_features
from emg_ai_arm.utils.config import WINDOWS_DIR, MODELS_DIR

DATA_PATH = WINDOWS_DIR / "fake_windows.npz"
MODEL_PATH = MODELS_DIR / "rf_model.joblib"

def main():
    data = np.load(str(DATA_PATH))
    Xw = data["X"]  # (N, SAMPLES, 2)
    y = data["y"]   # (N,)

    # Convert windows to feature vectors
    X = np.vstack([extract_features(w) for w in Xw])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, str(MODEL_PATH))
    print("Saved model:", MODEL_PATH)

if __name__ == "__main__":
    main()
