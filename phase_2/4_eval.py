import importlib.util
import torch
import numpy as np
import pandas as pd
import json
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

def prepare_tensors(df, device):
    vec_series = df['news_vector'].apply(parse_vector)
    X = torch.from_numpy(np.stack(vec_series.values)).to(device)
    if 'impact_score' in df.columns:
        y = torch.tensor(df['impact_score'].fillna(0).astype(int) + 3, dtype=torch.long).to(device)
    else:
        y = torch.zeros(len(df), dtype=torch.long).to(device)
    return X, y

def evaluate_model(model, X, y, batch_size=64):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total if total > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        X_test, y_test = prepare_tensors(test_df, device)
        # Evaluate
        acc = evaluate_model(model, X_test, y_test, batch_size=config.batch_size)
        print(f"Test accuracy for {name}: {acc*100:.2f}%")
        results[name] = acc
    print("\nSummary:")
    for name, acc in results.items():
        print(f"{name:>8}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
