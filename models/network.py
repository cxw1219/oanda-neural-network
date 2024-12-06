import torch
import torch.nn as nn
from config.config import ModelConfig

class GoldPriceNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super(GoldPriceNet, self).__init__()
        layers = []
        
        # Build network layers dynamically from config
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_number_of_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)