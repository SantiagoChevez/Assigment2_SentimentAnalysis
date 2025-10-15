import importlib.util
import sys
from pathlib import Path
import torch
import ast
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json

# Resolve important paths relative to this file to be robust to CWD
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

# Load the module dynamically from absolute paths
spec = importlib.util.spec_from_file_location("model", str(THIS_DIR / "2_model.py"))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
spec2 = importlib.util.spec_from_file_location("preprocess", str(THIS_DIR / "1_preprocess.py"))
pre_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pre_module)

def parse_vector(cell):
	"""Parse stringified vector back to numpy array."""
	try:
		return np.array(ast.literal_eval(cell), dtype=np.float32)
	except Exception:
		return np.array([], dtype=np.float32)

def load_and_preprocess_data(csv_path, start_date="2009-1-1"):
	"""Load and split dataset into train/test."""
	print(f"\nLoading data from: {csv_path}")
	train_df, test_df = pre_module.preprocess(str(csv_path), start_date)
	print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
	return train_df, test_df

def prepare_tensors(train_df, test_df, device):
	"""Parse news_vector and prepare X,y tensors."""
	if 'news_vector' not in train_df.columns:
		raise ValueError("There is no 'news_vector' column in the dataset.")

	train_vec_series = train_df['news_vector'].apply(parse_vector)
	test_vec_series  = test_df['news_vector'].apply(parse_vector)

	if train_vec_series.empty:
		raise ValueError("No lines in train_df after split.")

	first_len = len(train_vec_series.iloc[0])
	if first_len == 0:
		raise ValueError("The first vector is empty; failed to parse 'news_vector'.")

	bad_train = [i for i,v in enumerate(train_vec_series) if len(v) != first_len]
	bad_test  = [i for i,v in enumerate(test_vec_series) if len(v) != first_len]
	if bad_train:
		raise ValueError(f"Inconsistent vectors in train (example indices {bad_train[:5]}).")
	if bad_test:
		raise ValueError(f"Inconsistent vectors in test (example indices {bad_test[:5]}).")

	X_train_np = np.stack(train_vec_series.values)
	X_test_np  = np.stack(test_vec_series.values)
	input_dim = X_train_np.shape[1]

	# Process labels
	if 'impact_score' in train_df.columns:
		if train_df['impact_score'].isna().all():
			print("WARNING: All impact_score are NaN; cannot train yet.")
			y_train = torch.zeros(len(train_df), dtype=torch.long)
			y_test = torch.zeros(len(test_df), dtype=torch.long)
		else:
			# Map [-3,3] to [0,6]
			mapped_train = train_df['impact_score'].dropna().astype(int) + 3
			mapped_test = test_df['impact_score'].dropna().astype(int) + 3
			y_train = torch.tensor(mapped_train.reindex(train_df.index, fill_value=0).values, dtype=torch.long)
			y_test = torch.tensor(mapped_test.reindex(test_df.index, fill_value=0).values, dtype=torch.long)
	else:
		print("WARNING: No 'impact_score' column exists; creating dummy labels.")
		y_train = torch.zeros(len(train_df), dtype=torch.long)
		y_test = torch.zeros(len(test_df), dtype=torch.long)

	X_train = torch.from_numpy(X_train_np).to(device)
	X_test = torch.from_numpy(X_test_np).to(device)
	
	print(f"Training data shape: {X_train.shape}; Labels shape: {y_train.shape}")
	return X_train, y_train, X_test, y_test, input_dim

class NewsDataset(Dataset):
	def __init__(self, X, y):
		self.X = X  
		self.y = y
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]
# ===================== Training Loop =====================


def train_model(X_train, y_train, X_test, y_test, config, model_name="dtm"):
	"""Complete training pipeline with model creation, training loop, and saving."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config.device = str(device)
	
	# Create model
	model = model_module.MLP(config).to(device)
	print(f"\n=== {model_name.upper()} Model Training ===")
	print(f"Model has {model.get_num_parameters():,} trainable parameters (input_dim={config.input_dim})")
	print(f"Using device: {device}")

	# Forward test
	with torch.no_grad():
		out = model(X_train[:8])
	print(f"Forward OK -> output shape: {out.shape}")


	# Create DataLoaders
	train_dataset = NewsDataset(X_train, y_train.to(device))
	test_dataset = NewsDataset(X_test, y_test.to(device))

	train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

	# Select optimizer based on config
	if hasattr(config, 'optimizer'):
		opt_name = config.optimizer.lower()
	else:
		opt_name = 'adam'

	if opt_name == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	elif opt_name == 'adamw':
		optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	elif opt_name == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	else:
		raise ValueError(f"Unknown optimizer: {opt_name}")

	criterion = torch.nn.CrossEntropyLoss()

	print(f"Starting training for {config.epochs} epochs...")
	print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

	# Training loop
	for epoch in range(1, config.epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		correct_train = 0
		total_train = 0
		
		for batch_idx, (data, target) in enumerate(train_loader):
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item() * data.size(0)
			pred = output.argmax(dim=1, keepdim=True)
			correct_train += pred.eq(target.view_as(pred)).sum().item()
			total_train += target.size(0)
		
		avg_loss = total_loss / len(train_loader.dataset)
		train_acc = correct_train / total_train
		
		# Evaluation phase
		model.eval()
		test_loss = 0.0
		correct_test = 0
		total_test = 0
		
		with torch.no_grad():
			for data, target in test_loader:
				output = model(data)
				test_loss += criterion(output, target).item() * data.size(0)
				pred = output.argmax(dim=1, keepdim=True)
				correct_test += pred.eq(target.view_as(pred)).sum().item()
				total_test += target.size(0)
		
		test_loss /= len(test_loader.dataset)
		test_acc = correct_test / total_test
		
		# Print progress
		if epoch % 10 == 0 or epoch == 1:
			print(f'Epoch {epoch:3d}/{config.epochs} | '
				f'Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | '
				f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}')

	# Save the trained model
	models_dir = THIS_DIR / "models"
	os.makedirs(models_dir, exist_ok=True)
	model_save_path = models_dir / f"{model_name}_mlp_classifier.pth"
	torch.save(model.state_dict(), str(model_save_path))
	print(f"\nModel saved to: {model_save_path}")
	print(f"Final Test Accuracy: {test_acc:.3f}")

	# Save config for later loading
	config_dict = {
		"input_dim": config.input_dim,
		"output_dim": config.output_dim,
		"hidden_layers": config.hidden_layers,
		"activation": config.activation,
		"dropout": config.dropout
	}
	config_path = models_dir / f"{model_name}_mlp_config.json"
	with open(config_path, "w") as f:
		json.dump(config_dict, f, indent=2)
	print(f"Model config saved to: {config_path}")
	
	return model, test_acc

# ===================== MAIN TRAINING PIPELINE =====================

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Dataset configurations: (csv_path, config_function, model_name)
	DATASETS_DIR = ROOT_DIR / "datasets"
	datasets = [
		(DATASETS_DIR / "vectorized_news_dtm.csv", model_module.create_dtm_model_config, "dtm"),
		(DATASETS_DIR / "vectorized_news_tfidf.csv", model_module.create_tfidf_model_config, "tfidf"),  
		(DATASETS_DIR / "vectorized_news_curated.csv", model_module.create_curated_model_config, "curated")
	]
	
	results = {}
	
	for csv_path, config_fn, model_name in datasets:
		try:
			print(f"\n{'='*60}")
			print(f"TRAINING {model_name.upper()} CLASSIFIER")
			print(f"{'='*60}")
			
			# Load and preprocess data
			train_df, test_df = load_and_preprocess_data(csv_path)
			
			# Prepare tensors
			X_train, y_train, X_test, y_test, input_dim = prepare_tensors(train_df, test_df, device)
			
			# Create config with detected input_dim
			config = config_fn(input_dim=input_dim, num_classes=7)
			
			# Train model
			model, final_acc = train_model(X_train, y_train, X_test, y_test, config, model_name)
			
			results[model_name] = final_acc
			
		except Exception as e:
			print(f"ERROR training {model_name}: {e}")
			results[model_name] = 0.0
	
	# Summary
	print(f"\n{'='*60}")
	print("TRAINING SUMMARY")
	print(f"{'='*60}")
	for model_name, acc in results.items():
		print(f"{model_name.upper():>12}: {acc:.3f} accuracy")
	print(f"{'='*60}")