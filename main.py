import logging
from datetime import datetime
import time
import sys
import torch
import argparse

from config.config import load_config
from trading.trader import PriceTrader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Gold Price Trading Bot')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'trade'], default='train', help='Operation mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Load configurations
        logger.info("Loading configuration...")
        model_config, oanda_config, data_config = load_config(args.config)

        # Initialize trader
        logger.info("Initializing trader...")
        trader = PriceTrader(model_config, oanda_config, data_config)

        if args.mode == 'train':
            logger.info("Starting model training...")
            metrics = trader.train_model()
            logger.info(f"Training completed with metrics: {metrics}")
            
        elif args.mode == 'trade':
            logger.info("Starting real-time trading...")
            price_buffer = []
            
            try:
                for timestamp, prices in trader.stream_prices():
                    logger.info(f"Received prices at {timestamp}: {prices}")
                    
                    # Process streaming data and update predictions
                    price_buffer.append(prices)
                    if len(price_buffer) > model_config.buffer_size:
                        price_buffer.pop(0)
                        
                        # Make prediction here
                        # Note: Additional data processing would be needed to convert
                        # streaming prices into the same format as training data
                        
                    time.sleep(model_config.update_interval)

            except KeyboardInterrupt:
                logger.info("Stopping bot gracefully...")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()