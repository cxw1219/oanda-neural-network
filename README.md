# OANDA Neural Network Gold Price Predictor

Neural network model for predicting gold prices using OANDA data and a synthetic USDX index. This project uses PyTorch with DirectML for GPU acceleration on AMD graphics cards.

## Features

- Custom USDX index calculation from major currency pairs
- GPU acceleration using DirectML (supports AMD GPUs)
- Real-time data processing and prediction
- Comprehensive metric tracking and visualization
- Early stopping and model checkpointing

## Requirements

- Python 3.10+
- PyTorch with DirectML support
- OANDA API access
- AMD GPU (for DirectML acceleration)

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Create config.ini from template:
```bash
cp config.ini.template config.ini
```

3. Update config.ini with your:
   - OANDA API credentials
   - Data path
   - Model parameters

## Usage

Train the model:
```bash
python main.py
```

The script will:
1. Load and process historical data
2. Calculate synthetic USDX
3. Train the model using GPU acceleration
4. Generate visualizations of results

## Project Structure

```
oanda-neural-network/
├── config/
│   ├── __init__.py
│   └── config.py         # Configuration management
├── data/
│   ├── __init__.py
│   └── processor.py      # Data processing and USDX calculation
├── models/
│   ├── __init__.py
│   └── network.py        # Neural network architecture
├── utils/
│   ├── __init__.py
│   └── metrics.py        # Performance metrics
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
└── config.ini.template   # Configuration template
```

## Visualizations

The model generates:
- predictions.png: Shows predicted vs actual values
- training_history.png: Shows loss and accuracy metrics

## GPU Acceleration

The project uses torch-directml for GPU acceleration, particularly optimized for AMD graphics cards. Memory usage is optimized for efficient batch processing.