import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import logging
from datetime import datetime
from typing import Dict, Iterator, Tuple, Optional
import matplotlib.pyplot as plt
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream

from config.config import ModelConfig, OANDAConfig, DataConfig
from models.network import GoldPriceNet
from data.processor import DataProcessor
from utils.metrics import calculate_metrics, evaluate_model, log_metrics

logger = logging.getLogger(__name__)

class PriceTrader:
    def __init__(self, model_config: ModelConfig, oanda_config: OANDAConfig, data_config: DataConfig):
        self.config = model_config
        self.oanda_config = oanda_config
        self.data_config = data_config
        self.device = torch_directml.device()
        logger.info(f"Using DirectML device: {self.device}")
        
        # Initialize model and optimizer
        self.model = GoldPriceNet(model_config).to(self.device)
        logger.info(f"Model architecture: {self.model.network}")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        
        # Initialize data processor
        self.data_processor = DataProcessor(oanda_config, data_config, self.device)
        
        self.setup_plotting()

    def setup_plotting(self):
        """Initialize real-time plotting."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.predicted_line, = self.ax.plot([], [], 'b-', label="Predicted")
        self.actual_line, = self.ax.plot([], [], 'r-', label="Actual")
        self.ax.set_title("Gold Price Prediction")
        self.ax.set_xlabel("Time Steps")
        self.ax.set_ylabel("Price")
        plt.legend()
        plt.grid(True)

    def train_model(self) -> Dict[str, float]:
        """Train model on historical data."""
        historical_data = self.data_processor.load_historical_data()
        train_loader, val_loader = self.data_processor.prepare_training_data(historical_data)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(10):  # 10 epochs
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = evaluate_model(self.model, val_loader, self.criterion, self.device)
            
            # Early stopping check
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1

            # Log progress
            log_metrics(train_metrics, "Train ")
            log_metrics(val_metrics, "Validation ")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self._load_best_checkpoint()
        return val_metrics

    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0
        batch_count = 0
        
        for batch_data, batch_targets in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_data)
            loss = self.criterion(predictions, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1

        return {
            'avg_loss': epoch_loss / batch_count
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        logger.info("Saving new best model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, 'best_model.pth')

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model checkpoint")

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model."""
        self.model.eval()
        with torch.no_grad():
            return self.model(data)

    def stream_prices(self) -> Iterator[Tuple[datetime, Dict[str, float]]]:
        """Stream real-time prices from OANDA."""
        api = API(
            access_token=self.oanda_config.access_token,
            environment=self.oanda_config.environment
        )
        
        params = {"instruments": ",".join(self.oanda_config.instruments)}
        pricing_stream = PricingStream(accountID=self.oanda_config.account_id, params=params)
        
        try:
            for price in api.request(pricing_stream):
                if price["type"] == "PRICE":
                    timestamp = datetime.strptime(price["time"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    instrument = price["instrument"]
                    current_price = float(price["bids"][0]["price"])
                    
                    yield timestamp, {instrument: current_price}
                    
        except Exception as e:
            logger.error(f"Error in price streaming: {str(e)}")
            raise

    def update_plot(self, predictions: torch.Tensor, actuals: torch.Tensor, window: int = 100):
        """Update real-time visualization."""
        self.ax.clear()
        n = len(predictions)
        if n > window:
            start_idx = n - window
        else:
            start_idx = 0
            
        x = range(n-start_idx)
        self.predicted_line.set_data(x, predictions[start_idx:].cpu().numpy())
        self.actual_line.set_data(x, actuals[start_idx:].cpu().numpy())
        
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)