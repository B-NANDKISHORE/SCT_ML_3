import os
import json
import argparse
import joblib
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CLASS_NAMES = ["cats", "dogs"]

def load_folder_images(root_dir, img_size=64, max_per_class=None):
    X, y = [], []
    for label, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(root_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        files = [f for f in os.listdir(folder) if not f.startswith(".")]
        if max_per_class:
            files = files[:max_per_class]
        for f in tqdm(files, desc=f"Loading {cls}"):
            path = os.path.join(folder, f)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype(np.float32) / 255.0
                X.append(img.flatten())
                y.append(label)
            except Exception:
                continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join("data"),
                    help="Path with two folders: data/cats and data/dogs")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--pca", type=int, default=256,
                    help="Number of PCA components (0 disables PCA)")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="Optionally cap samples per class for quick tests")
    ap.add_argument("--out", default="svm_catsdogs.joblib")
    args = ap.parse_args()

    print("Loading data from:", os.path.abspath(args.data_dir))
    X, y = load_folder_images(args.data_dir, img_size=args.img_size,
                              max_per_class=args.max_per_class)
    if len(X) == 0:
        raise RuntimeError("No images loaded. Check folder names and contents.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    steps = [("scaler", StandardScaler())]
    if args.pca and args.pca > 0:
        steps.append(("pca", PCA(n_components=min(args.pca, X_train.shape[1]))))
    steps.append(("svm", SVC(kernel=args.kernel, C=args.C, gamma=args.gamma, probability=True)))
    pipe = Pipeline(steps)

    print("Training SVM...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump({
        "pipeline": pipe,
        "img_size": args.img_size,
        "class_names": CLASS_NAMES
    }, args.out)

    meta = {
        "data_dir": os.path.abspath(args.data_dir),
        "img_size": args.img_size,
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "pca_components": args.pca,
        "accuracy": acc
    }
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved model ->", args.out)

if __name__ == "__main__":
    main()
