import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate trading metrics from predictions."""
    with torch.no_grad():
        # Mean Absolute Error
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # Mean Squared Error
        mse = torch.mean((predictions - targets) ** 2).item()
        
        # Direction Accuracy
        if predictions.size(0) > 1:
            pred_direction = (predictions[1:] > predictions[:-1]).float()
            true_direction = (targets[1:] > targets[:-1]).float()
            direction_accuracy = torch.mean((pred_direction == true_direction).float()).item() * 100
        else:
            direction_accuracy = 0.0

        # Maximum Drawdown
        cum_returns = torch.cumsum(targets[1:] - targets[:-1], dim=0)
        running_max = torch.maximum.accumulate(cum_returns)
        drawdowns = running_max - cum_returns
        max_drawdown = torch.max(drawdowns).item() if len(drawdowns) > 0 else 0.0
        
        return {
            'mae': mae,
            'mse': mse,
            'direction_accuracy': direction_accuracy,
            'max_drawdown': max_drawdown
        }

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance on given dataset."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    batch_count = 0

    with torch.no_grad():
        for batch_data, batch_targets in data_loader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = model(batch_data)
            loss = criterion(predictions, batch_targets)
            
            total_loss += loss.item()
            batch_count += 1
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_targets.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics['avg_loss'] = total_loss / batch_count
    
    return metrics

def calculate_sharpe_ratio(
    returns: torch.Tensor, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio."""
    returns = returns.cpu().numpy()
    excess_returns = returns - risk_free_rate/periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino_ratio(
    returns: torch.Tensor,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio using only negative returns for denominator."""
    returns = returns.cpu().numpy()
    excess_returns = returns - risk_free_rate/periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)

def log_metrics(metrics: Dict[str, float], prefix: str = '') -> None:
    """Log metrics with optional prefix."""
    for name, value in metrics.items():
        logger.info(f"{prefix}{name}: {value:.4f}")