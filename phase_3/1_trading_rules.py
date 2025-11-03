import math
from pathlib import Path
import pandas as pd
import numpy as np
import importlib.util, json
import torch
import ast

# Resolve base directories independent of current working directory
from xml.parsers.expat import model

# Load model module dynamically 
BASE_DIR = Path(__file__).resolve().parent  # repo root
REPO_ROOT = BASE_DIR.parent    
PHASE2 = REPO_ROOT / "phase_2"                          # .../Assigment2_SentimentAnalysis
MODELS_DIR = PHASE2 / "models"
DATASETS_DIR = REPO_ROOT / "datasets"

spec = importlib.util.spec_from_file_location("model_module", str(PHASE2 / "2_model.py"))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

def load_trained_model(model_key="tfidf"):
    """
    model_key ∈ {"dtm","tfidf","curated"}
    Returns: (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = MODELS_DIR
    config_path = models_dir / f"{model_key}_mlp_config.json"
    weights_path = models_dir / f"{model_key}_mlp_classifier_best.pth"

    cfg_dict = json.loads(Path(config_path).read_text())
    # Recreate architecture using saved shapes
    # Map to your factory function:
    if model_key == "dtm":
        cfg = model_module.create_dtm_model_config(
            input_dim=cfg_dict["input_dim"],
            num_classes=cfg_dict["output_dim"]
        )
        dataset = DATASETS_DIR / "vectorized_news_dtm.csv"
    elif model_key == "tfidf":
        cfg = model_module.create_tfidf_model_config(
            input_dim=cfg_dict["input_dim"],
            num_classes=cfg_dict["output_dim"]
        )
        dataset = DATASETS_DIR / "vectorized_news_tfidf.csv"
    else:
        cfg = model_module.create_curated_model_config(
            input_dim=cfg_dict["input_dim"],
            num_classes=cfg_dict["output_dim"]
        )
        dataset = DATASETS_DIR / "vectorized_news_curated.csv"

    # Ensure activation/dropout match the saved config
    cfg.activation = cfg_dict.get("activation", cfg.activation)
    cfg.dropout = cfg_dict.get("dropout", cfg.dropout)

    model = model_module.MLP(cfg).to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    dataset = pd.read_csv(dataset)

    return model, device, dataset

def parse_vector(cell):
    """Parse stringified vector into numpy array."""
    try:
        return np.array(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)
        
def prepare_tensors(df, device):
    vec_series = df['news_vector'].apply(parse_vector)
    X = torch.from_numpy(np.stack(vec_series.values)).to(device)
    return X
        
def predict_impact_from_vectors(model, device, X):
    """
    vectors: np.ndarray of shape [N, input_dim], dtype float32
    Returns: dict with logits, probs, class_idx, impact_score
    """
    with torch.no_grad():
        logits = model(X)               # [N, C]
        probs = torch.softmax(logits, dim=-1)  # [N, C]
        idx = probs.argmax(dim=-1)      # [N]
        # Map predicted class index to a signed impact score.
        # For a 3-class model we want mapping: 0 -> -2, 1 -> 0, 2 -> 2
        # For other odd-numbered class counts, center at zero: e.g., 7 -> -3..3
        n_classes = logits.shape[1]
        if n_classes == 3:
            impact_tensor = (idx - 1) * 2
        else:
            offset = n_classes // 2
            impact_tensor = idx - offset
        impact = impact_tensor.cpu().numpy()
        return {"impact_score": impact}
    

def model_predict(model_key="tfidf"):
    """
    Takes a DataFrame with a 'news_vector' column (containing vectorized features)
    and returns the DataFrame with a new 'pred_impact_score' column.
    
    Args:
        df: DataFrame with 'news_vector' column containing feature vectors
        model_key: which model to use ("tfidf", "dtm", or "curated")
    
    Returns:
        DataFrame with added 'pred_impact_score' column (values in range [-3, 3])
    """

    # Load trained model
    model, device, dataset = load_trained_model(model_key=model_key)
    
    # load df
    df = dataset.copy()
    df.sort_values(['date', 'symbol'], inplace=True)

    #test with first 50 unique symbols
    df = df.groupby('symbol').first().head(50).reset_index()

    # Convert news_vector column to numpy array
    X = prepare_tensors(df, device) 
    
    # Get predictions
    predictions = predict_impact_from_vectors(model, device, X)
    
    # Add impact scores to dataframe
    #df = df.copy()  # Avoid modifying original
    df['pred_impact_score'] = predictions['impact_score']
    df.drop(columns=['news_vector'], inplace=True)
    
    return df

def calc_shares(stock):
    """Calculate number of shares to trade from stock dict.

    Expects stock to contain 'pred_impact_score', 'price' and 'balance'.
    Returns an integer >= 1.
    """
    alpha = 1
    # use built-in abs to get absolute value
    s = abs(stock.get('pred_impact_score', 0))
    price = float(stock.get('price', 0.0))
    balance = float(stock.get('balance', 0.0))
    if price <= 0 or balance <= 0:
        return 0
    shares = max(1, math.floor((alpha * s / 100) * balance / price))
    return int(shares)

def buy_rule(stock, balance, owned_shares):
    """Return number of shares to buy (int)."""
    # trading simulation passes balance separately; ensure stock has it for calc_shares
    stock = dict(stock)
    stock['balance'] = balance
    shares = calc_shares(stock)
    return int(shares)

def sell_rule(stock, balance, owned_shares):
    """Return number of shares to sell (int)."""
    stock = dict(stock)
    stock['balance'] = balance
    shares = calc_shares(stock)
    # cap to owned_shares
    shares = min(int(shares), int(owned_shares))
    return int(shares) if shares > 0 else 0

def trading_rules(stock, balance, owned_shares):
    """Return integer number of shares to buy (>0) or sell (>0), or None for no action.

    Positive decision is buy (simulator looks at pred_impact_score sign), negative
    is not used because simulator decides direction from pred_impact_score.
    We follow the simulator contract: return an int (number of shares) or None.
    """
    try:
        impact = float(stock.get('pred_impact_score', 0))
        price = float(stock.get('price', 0.0))
    except Exception:
        return None

    if impact > 0 and price > 0 and balance / price > 0:
        # buy path: return positive integer shares
        n = buy_rule(stock, balance, owned_shares)
        return int(n) if n > 0 else None
    elif impact < 0 and owned_shares > 0:
        # sell path: return positive integer shares to sell
        n = sell_rule(stock, balance, owned_shares)
        return int(n) if n > 0 else None
    else:
        return None
    
if __name__ == "__main__":
    df_is = model_predict(model_key="tfidf")
    print(df_is.head(10))
