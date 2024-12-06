from dataclasses import dataclass
from typing import List, Optional
import configparser
import os

@dataclass
class ModelConfig:
    input_dim: int = 4  # [gold_norm, gold_returns, usdx_norm, usdx_returns]
    hidden_dims: List[int] = None
    buffer_size: int = 60
    learning_rate: float = 0.001
    update_interval: int = 5
    batch_size: int = 64

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [16, 8]

@dataclass
class DataConfig:
    historical_data_path: str
    data_frequency: str = '1Min'  # Resample frequency
    training_size: int = 100000   # Number of samples to use for training

@dataclass
class OANDAConfig:
    account_id: str
    access_token: str
    environment: str = 'practice'
    instruments: List[str] = None

    def __post_init__(self):
        if self.instruments is None:
            self.instruments = [
                "XAU_USD", "EUR_USD", "USD_JPY", "GBP_USD",
                "USD_CAD", "USD_CHF", "USD_SEK"
            ]

def load_config(config_path: str = 'config.ini') -> tuple[ModelConfig, DataConfig, OANDAConfig]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    model_config = ModelConfig(
        input_dim=4,
        hidden_dims=[16, 8],
        buffer_size=config.getint('MODEL', 'buffer_size', fallback=60),
        learning_rate=config.getfloat('MODEL', 'learning_rate', fallback=0.001),
        update_interval=config.getint('MODEL', 'update_interval', fallback=5),
        batch_size=config.getint('MODEL', 'batch_size', fallback=64)
    )

    data_config = DataConfig(
        historical_data_path=config.get('DATA', 'historical_data_path'),
        data_frequency=config.get('DATA', 'data_frequency', fallback='1Min'),
        training_size=config.getint('DATA', 'training_size', fallback=100000)
    )

    oanda_config = OANDAConfig(
        account_id=config.get('OANDA', 'account_id'),
        access_token=config.get('OANDA', 'access_token'),
        environment=config.get('OANDA', 'environment', fallback='practice')
    )

    return model_config, data_config, oanda_config