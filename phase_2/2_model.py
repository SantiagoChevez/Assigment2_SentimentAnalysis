from dataclasses import dataclass
from typing import List, Optional, Literal
import torch
import torch.nn as nn

# Type hints for better code documentation
ActivationName = Literal["relu", "gelu", "tanh"]
OptimizerName = Literal["sgd", "adam", "adamw"]


@dataclass
class MLPConfig:
    """Configuration class for MLP hyperparameters and architecture."""
    
    # Architecture parameters
    input_dim: int                      
    output_dim: int                     
    hidden_layers: List[int]            
    activation: ActivationName = "relu" 
    dropout: float = 0.2                
    
    # Training hyperparameters
    learning_rate: float = 1e-3         
    batch_size: int = 64                
    epochs: int = 50                    
    optimizer: OptimizerName = "adam"   
    weight_decay: float = 1e-4                         

def get_activation(name) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for news sentiment classification.
    
    Architecture:
    - Input layer: Takes vectorized news features
    - Hidden layers: Configurable depth and width with activation functions
    - Dropout: Applied after each hidden layer for regularization
    - Output layer: Linear layer for classification logits
    
    Args:
        cfg (MLPConfig): Configuration object containing architecture and hyperparameters
    """
    
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.config = cfg
        layers = []
        in_dim = cfg.input_dim

        # Build hidden layers dynamically
        for i, hidden_size in enumerate(cfg.hidden_layers):
            
            layers.append(nn.Linear(in_dim, hidden_size))         
            layers.append(get_activation(cfg.activation))          
            if cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
            
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, cfg.output_dim))
        
        # Combine all layers
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def create_dtm_model_config(input_dim: int = 5000, num_classes: int = 7) -> MLPConfig:
    """
    Create configuration for DTM (Document-Term Matrix) classifier.
    
    """
    
    return MLPConfig(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_layers=[512, 256, 128],
        activation="relu",
        dropout=0.3,
        learning_rate=1e-3,
        batch_size=64,
        epochs=50,
        optimizer="adam",
        weight_decay=1e-4
    )


def create_tfidf_model_config(input_dim: int = 5000, num_classes: int = 7) -> MLPConfig:
    """
    Create configuration for TF-IDF classifier.
    
    """
    return MLPConfig(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_layers=[768, 512, 256, 128],
        activation="gelu",
        dropout=0.2,
        learning_rate=8e-4,  
        batch_size=64,
        epochs=60,
        optimizer="adamw",  
        weight_decay=1e-4
    )


def create_curated_model_config(input_dim: int = 100, num_classes: int = 7) -> MLPConfig:
    """
    Create configuration for Curated features classifier.
    
    """
    return MLPConfig(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_layers=[256, 128],
        activation="tanh",
        dropout=0.4,
        learning_rate=1e-3,
        batch_size=32,  
        epochs=40,
        optimizer="adam",
        weight_decay=5e-4  
    )
        


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Create configurations for all three model types
    configs = {
        "DTM": create_dtm_model_config(input_dim=5000, num_classes=7),
        "TF-IDF": create_tfidf_model_config(input_dim=5000, num_classes=7), 
        "Curated": create_curated_model_config(input_dim=200, num_classes=7)
    }
    
    # Display model architectures and create test instances
    for name, config in configs.items():
        print(f"\n{name} Model Configuration:")
        print(f"  Input dimension: {config.input_dim}")
        print(f"  Hidden layers: {config.hidden_layers}")
        print(f"  Activation: {config.activation}")
        print(f"  Dropout: {config.dropout}")
        print(f"  Output classes: {config.output_dim}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Optimizer: {config.optimizer}")
        
        # Create model and test forward pass
        config.device = device
        model = MLP(config).to(device)
        
        # Test with sample input
        batch_size = 8
        sample_input = torch.randn(batch_size, config.input_dim, device=device)
        
        with torch.no_grad():
            output = model(sample_input)
            
        print(f"  Parameters: {model.get_num_parameters():,}")
        print(f"  Test input shape: {sample_input.shape}")
        print(f"  Test output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print("-" * 50)
    