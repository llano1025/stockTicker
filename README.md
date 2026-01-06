# Stock Selection System

A comprehensive Python-based stock selection system implementing an 8-step filtering strategy with backtesting and continuous parameter optimization capabilities.

## Overview

This system automates the process of selecting stocks based on technical and fundamental criteria, including:
- Daily gain filtering (3%-5%)
- Volume ratio analysis
- Turnover rate filtering
- Market capitalization constraints
- Volume trend analysis
- Moving average alignment
- Market strength comparison
- New high detection

## Features

- **8-Step Stock Selection**: Implements a rigorous filtering process to identify high-potential stocks
- **Backtesting Engine**: Test strategy performance on historical data
- **Parameter Optimization**: Automatically find optimal parameters using grid search, random search, or Bayesian optimization
- **Modular Architecture**: Clean, maintainable code structure with separate components
- **Comprehensive Logging**: Track all operations and results
- **Visual Analytics**: Generate performance charts and reports

## Installation

1. Clone the repository:
```bash
cd /home/llanopi/git/stockTicker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize parameters:

```yaml
# Key parameters you can adjust:
gain_range:
  min: 3.0  # Minimum daily gain percentage
  max: 5.0  # Maximum daily gain percentage

volume_ratio:
  min: 1.0  # Minimum volume ratio

market_cap:
  min: 50.0   # Minimum market cap (billions)
  max: 200.0  # Maximum market cap (billions)

# And many more...
```

## Usage

### 1. Stock Selection Mode (Real-time)

Select stocks based on current market data:

```bash
python main.py --mode select
```

This will:
- Fetch latest stock data
- Apply 8-step filtering
- Display selected stocks with entry signals
- Show selection funnel statistics

### 2. Backtest Mode

Test strategy performance on historical data:

```bash
python main.py --mode backtest
```

This will:
- Run strategy on historical data (date range from config)
- Execute trades with stop-loss and take-profit
- Calculate performance metrics (Sharpe ratio, win rate, etc.)
- Generate performance charts
- Save results to CSV files

**Output includes:**
- Total return
- Win rate
- Sharpe ratio
- Maximum drawdown
- Trade-by-trade breakdown

### 3. Optimization Mode

Find optimal parameters for the strategy:

```bash
python main.py --mode optimize
```

This will:
- Test multiple parameter combinations
- Evaluate each using backtesting
- Identify best-performing parameters
- Save results and updated configuration

**Optimization methods:**
- Grid Search: Test all combinations
- Random Search: Sample random combinations
- Bayesian: Intelligent parameter search (planned)

## System Architecture

```
stockTicker/
├── main.py              # Main entry point and orchestration
├── config.yaml          # Configuration parameters
├── data_fetcher.py      # Stock data retrieval and caching
├── stock_selector.py    # 8-step selection logic
├── backtester.py        # Backtesting engine
├── optimizer.py         # Parameter optimization
├── requirements.txt     # Python dependencies
└── logs/               # Output logs and results
```

## 8-Step Selection Process

### Step 1: Gainers Filter (3%-5%)
Identifies stocks with daily gains between 3% and 5% - strong movers but not overextended.

### Step 2: Volume Ratio >= 1
Ensures above-average trading volume, indicating genuine interest.

### Step 3: Turnover Rate (5%-10%)
Filters for healthy liquidity without excessive speculation.

### Step 4: Market Cap ($50B-$200B)
Focuses on large-cap stocks with stability and liquidity.

### Step 5: Volume Trend
Keeps stocks with continuously increasing volume, showing building momentum.

### Step 6: MA Alignment
Selects stocks where short-term MA aligns with 60-day line pointing upward, indicating trend strength.

### Step 7: Market Strength
Retains only stocks outperforming the market benchmark (S&P 500).

### Step 8: New Highs
Final filter for stocks hitting new highs at end of day with MA support - the entry trigger.

## Backtesting Features

- **Position Management**: Automatic position sizing based on capital allocation
- **Risk Management**: Stop-loss (-5%) and take-profit (10%) orders
- **Hold Period**: Configurable holding period (default: 5 days)
- **Portfolio Tracking**: Real-time portfolio value calculation
- **Performance Metrics**:
  - Total return
  - Sharpe ratio
  - Win rate
  - Profit factor
  - Maximum drawdown
  - Average win/loss

## Optimization Features

- **Multiple Methods**: Grid search, random search, Bayesian (planned)
- **Configurable Metrics**: Optimize for Sharpe ratio, total return, or win rate
- **Parameter Analysis**: Identify most impactful parameters
- **Auto-save**: Best configurations saved automatically

## Output Files

All results are saved to the `logs/` directory:

- `stock_selector_YYYYMMDD_HHMMSS.log` - Execution log
- `backtest_trades_YYYYMMDD_HHMMSS.csv` - Individual trade records
- `backtest_portfolio_YYYYMMDD_HHMMSS.csv` - Portfolio values over time
- `backtest_plot_YYYYMMDD_HHMMSS.png` - Performance visualization
- `optimization_results_YYYYMMDD_HHMMSS.csv` - All tested parameter combinations
- `best_config_YYYYMMDD_HHMMSS.yaml` - Optimal configuration
- `parameter_importance_YYYYMMDD_HHMMSS.csv` - Parameter sensitivity analysis

## Example Results

```
BACKTEST RESULTS
================================================================================
Period: 2023-01-01 to 2024-12-31
Initial Capital: $100,000.00
Final Value: $125,430.00
Total Return: 25.43%
Total P&L: $25,430.00

Total Trades: 48
Winning Trades: 32
Losing Trades: 16
Win Rate: 66.67%

Average Win: 8.45%
Average Loss: -3.21%
Profit Factor: 2.63

Sharpe Ratio: 1.82
Maximum Drawdown: -12.34%
================================================================================
```

## Customization

### Adding Custom Tickers

Edit `data_fetcher.py` to add your custom ticker list:

```python
def get_ticker_list(self, source: str = "nasdaq") -> List[str]:
    if source == "custom":
        tickers = ["AAPL", "MSFT", "GOOGL", ...]  # Your tickers
    return tickers
```

### Adjusting Risk Parameters

Edit `config.yaml`:

```yaml
backtest:
  position_size: 0.1    # 10% of capital per position
  max_positions: 10     # Maximum concurrent positions
  stop_loss: -0.05      # -5% stop loss
  take_profit: 0.10     # 10% take profit
  hold_days: 5          # Days to hold
```

### Custom Optimization Parameters

Edit `config.yaml` optimization section:

```yaml
optimization:
  enabled: true
  method: "grid_search"
  metrics: "sharpe_ratio"
  parameters:
    gain_range_min: [2.0, 3.0, 4.0]
    gain_range_max: [4.0, 5.0, 6.0]
    # Add more parameters to optimize
```

## Advanced Usage

### Using Custom Configuration

```bash
python main.py --mode backtest --config my_custom_config.yaml
```

### Programmatic Usage

```python
from data_fetcher import DataFetcher
from stock_selector import StockSelector
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize
data_fetcher = DataFetcher(config)
selector = StockSelector(config, data_fetcher)

# Get tickers
tickers = data_fetcher.get_ticker_list('nasdaq')

# Select stocks
selected = selector.select_stocks(tickers)

# Process results
for stock in selected:
    print(f"{stock['ticker']}: {stock['gain']:.2f}% gain")
```

## Performance Considerations

- **Data Caching**: Stock data is cached to reduce API calls
- **Batch Processing**: Optimization uses batch processing for efficiency
- **Logging Levels**: Adjust logging level in config for performance:
  - DEBUG: Detailed information (slower)
  - INFO: Standard output (recommended)
  - WARNING: Errors only (faster)

## Troubleshooting

### No Data Retrieved
- Check internet connection
- Verify ticker symbols are valid
- Check yfinance API status

### Slow Performance
- Reduce number of tickers
- Increase logging level to WARNING
- Use shorter date ranges for backtesting

### No Stocks Selected
- Review selection criteria in config
- Check if parameters are too restrictive
- Verify market conditions align with strategy

## Limitations

- Historical data quality depends on yfinance
- Backtesting assumes perfect execution (no slippage)
- Does not account for transaction costs
- Simplified turnover rate calculation (when shares outstanding unavailable)

## Future Enhancements

- Real-time data streaming
- Multi-timeframe analysis
- Machine learning integration
- Portfolio rebalancing strategies
- Advanced risk metrics (VaR, CVaR)
- Web dashboard interface
- Email/SMS alerts for signals

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## Disclaimer

This system is for educational and research purposes only. Always do your own research and consult with a financial advisor before making investment decisions. Past performance does not guarantee future results.

## License

MIT License - Feel free to use and modify as needed.

## Support

For questions or issues, please check the logs in the `logs/` directory for detailed error information.
