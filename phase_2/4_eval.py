import importlib.util
import sys
import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import ast

# Resolve base directories independent of current working directory
BASE_DIR = Path(__file__).resolve().parent               # .../phase_2
REPO_ROOT = BASE_DIR.parent                              # .../Assigment2_SentimentAnalysis
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = REPO_ROOT / "datasets"

# Dynamic import of model and preprocess modules using absolute paths
model_py = BASE_DIR / "2_model.py"
preprocess_py = BASE_DIR / "1_preprocess.py"

spec = importlib.util.spec_from_file_location("model_module", str(model_py))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

spec2 = importlib.util.spec_from_file_location("preprocess", str(preprocess_py))
pre_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pre_module)

def parse_vector(cell):
    """Safely parse a stringified list like "[0.1, 0.2, ...]" into a float32 numpy array."""
    try:
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def prepare_tensors(df, device, three_class=False):
    vec_series = df['news_vector'].apply(parse_vector)
    X = torch.from_numpy(np.stack(vec_series.values)).to(device)
    if 'impact_score' in df.columns:
        y = torch.tensor(df['impact_score'].fillna(0).astype(int) + 3, dtype=torch.long)
    else:
        y = torch.zeros(len(df), dtype=torch.long)

    if three_class:
        # map 0..6 -> 0/1/2
        y_cpu = y
        three = torch.zeros_like(y_cpu)
        three[y_cpu < 3] = 0
        three[y_cpu == 3] = 1
        three[y_cpu > 3] = 2
        y = three

    return X, y.to(device)

def evaluate_model(model, X, y, batch_size=64):
    """Run model on X and return (accuracy, preds_array, true_array)."""
    model.eval()
    preds_all = []
    true_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            true_all.append(batch_y.cpu().numpy())

    if len(preds_all) == 0:
        return 0.0, np.array([]), np.array([])

    preds_arr = np.concatenate(preds_all)
    true_arr = np.concatenate(true_all)
    acc = (preds_arr == true_arr).sum() / len(true_arr)
    return acc, preds_arr, true_arr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # default to 3-class mapping; pass --seven or --seven_class to use 7 classes
    three_class = not ('--seven_class' in sys.argv or '--seven' in sys.argv)
    configs = [
        (
            DATASETS_DIR / "vectorized_news_dtm.csv",
            MODELS_DIR / "dtm_mlp_classifier.pth",
            MODELS_DIR / "dtm_mlp_config.json",
            model_module.create_dtm_model_config,
            "DTM",
        ),
        (
            DATASETS_DIR / "vectorized_news_tfidf.csv",
            MODELS_DIR / "tfidf_mlp_classifier.pth",
            MODELS_DIR / "tfidf_mlp_config.json",
            model_module.create_tfidf_model_config,
            "TFIDF",
        ),
        (
            DATASETS_DIR / "vectorized_news_curated.csv",
            MODELS_DIR / "curated_mlp_classifier.pth",
            MODELS_DIR / "curated_mlp_config.json",
            model_module.create_curated_model_config,
            "CURATED",
        ),
    ]
    results = {}
    for csv_path, model_path, config_path, config_fn, name in configs:
        print(f"\nEvaluating {name} model...")
        # Load config
        with open(str(config_path), "r") as f:
            config_dict = json.load(f)
        config = config_fn(input_dim=config_dict["input_dim"], num_classes=config_dict["output_dim"])
        # Load model
        model = model_module.MLP(config).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        # Load and split data
        train_df, test_df = pre_module.preprocess(str(csv_path), "2009-1-1")
        # class distribution
        if not three_class:
            print("Class distribution (train):")
            print(train_df['impact_score'].fillna(0).astype(int).add(3).value_counts().sort_index())
            print("Class distribution (test):")
            print(test_df['impact_score'].fillna(0).astype(int).add(3).value_counts().sort_index())
        else:
            # show 3-class distribution
            remap = lambda s: s.fillna(0).astype(int).apply(lambda v: 0 if v<0 else (1 if v==0 else 2))
            print("3-class distribution (train):")
            print(remap(train_df['impact_score']).value_counts().sort_index())
            print("3-class distribution (test):")
            print(remap(test_df['impact_score']).value_counts().sort_index())

        X_test, y_test = prepare_tensors(test_df, device, three_class=three_class)
        # Evaluate and collect preds
        acc, preds_arr, true_arr = evaluate_model(model, X_test, y_test, batch_size=config.batch_size)
        print(f"Test accuracy for {name}: {acc*100:.2f}%")

        # Print classification report and confusion matrix (on CPU arrays)
        if len(true_arr) > 0:
            labels = sorted(list(set(true_arr.tolist())))
            print("\nClassification report:")
            print(classification_report(true_arr, preds_arr, labels=labels, digits=4, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(true_arr, preds_arr, labels=labels))
        else:
            print("No test labels to report detailed metrics.")

        results[name] = acc
    print("\nSummary:")
    for name, acc in results.items():
        print(f"{name:>8}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
