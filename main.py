"""
Main entry point for stock selection system
"""
import yaml
import logging
import argparse
import os
from datetime import datetime
from data_fetcher import DataFetcher
from stock_selector import StockSelector
from backtester import Backtester
from optimizer import StrategyOptimizer


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_dir = log_config.get('output_dir', 'logs')

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Setup logging
    log_file = os.path.join(log_dir, f'stock_selector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def mode_select_stocks(config: dict):
    """
    Mode: Select stocks for today/latest data

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("MODE: Stock Selection")
    logger.info("=" * 80)

    # Initialize components
    data_fetcher = DataFetcher(config)
    selector = StockSelector(config, data_fetcher)

    # Get tickers
    tickers_source = config.get('data', {}).get('tickers_source', 'nasdaq')
    tickers = data_fetcher.get_ticker_list(tickers_source)
    logger.info(f"Analyzing {len(tickers)} stocks from {tickers_source}")

    # Select stocks
    selected_stocks = selector.select_stocks(tickers)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("SELECTED STOCKS")
    logger.info("=" * 80)

    if not selected_stocks:
        logger.info("No stocks met all criteria")
    else:
        logger.info(f"Found {len(selected_stocks)} stocks meeting all criteria:\n")

        for i, stock in enumerate(selected_stocks, 1):
            logger.info(f"{i}. {stock['ticker']}")
            logger.info(f"   Close: ${stock['close']:.2f}")
            logger.info(f"   Daily Gain: {stock['gain']:.2f}%")
            logger.info(f"   Volume Ratio: {stock['volume_ratio']:.2f}")
            logger.info(f"   MA5: ${stock['ma5']:.2f} | MA20: ${stock['ma20']:.2f} | MA60: ${stock['ma60']:.2f}")
            logger.info(f"   New High: {stock['is_new_high']}")
            logger.info(f"   Entry Signal: {'YES' if stock['entry_signal'] else 'NO'}")
            logger.info("")

    # Display selection statistics
    stats = selector.get_selection_stats()
    logger.info("=" * 80)
    logger.info("SELECTION FUNNEL")
    logger.info("=" * 80)
    logger.info(f"Step 1 - Gainers (3%-5%):           {stats['step1_gainers']}")
    logger.info(f"Step 2 - Volume Ratio >= 1:         {stats['step2_volume_ratio']}")
    logger.info(f"Step 3 - Turnover Rate 5%-10%:      {stats['step3_turnover_rate']}")
    logger.info(f"Step 4 - Market Cap $50B-$200B:     {stats['step4_market_cap']}")
    logger.info(f"Step 5 - Volume Trend Increasing:   {stats['step5_volume_trend']}")
    logger.info(f"Step 6 - MA Alignment:              {stats['step6_ma_alignment']}")
    logger.info(f"Step 7 - Stronger than Market:      {stats['step7_market_strength']}")
    logger.info(f"Step 8 - New Highs:                 {stats['step8_new_highs']}")
    logger.info(f"FINAL - Selected Stocks:            {stats['final_selected']}")
    logger.info("=" * 80)


def mode_backtest(config: dict):
    """
    Mode: Run backtest on historical data

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("MODE: Backtesting")
    logger.info("=" * 80)

    # Initialize components
    data_fetcher = DataFetcher(config)
    selector = StockSelector(config, data_fetcher)
    backtester = Backtester(config, data_fetcher, selector)

    # Get parameters
    backtest_config = config.get('backtest', {})
    start_date = backtest_config.get('start_date', '2023-01-01')
    end_date = backtest_config.get('end_date', '2024-12-31')

    # Get tickers
    tickers_source = config.get('data', {}).get('tickers_source', 'nasdaq')
    tickers = data_fetcher.get_ticker_list(tickers_source)
    logger.info(f"Backtesting with {len(tickers)} stocks from {tickers_source}")

    # Run backtest
    results = backtester.run_backtest(start_date, end_date, tickers)

    # Display results
    metrics = results['metrics']

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    logger.info(f"Final Value: ${metrics['final_value']:,.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    logger.info("")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Winning Trades: {metrics['winning_trades']}")
    logger.info(f"Losing Trades: {metrics['losing_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info("")
    logger.info(f"Average Win: {metrics['avg_win']:.2f}%")
    logger.info(f"Average Loss: {metrics['avg_loss']:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info("")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info("=" * 80)

    # Save results if configured
    if config.get('logging', {}).get('save_results', True):
        log_dir = config.get('logging', {}).get('output_dir', 'logs')

        # Save trades
        trades_df = backtester.get_results_dataframe()
        trades_file = os.path.join(log_dir, f'backtest_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Trades saved to: {trades_file}")

        # Save portfolio values
        portfolio_df = backtester.get_portfolio_dataframe()
        portfolio_file = os.path.join(log_dir, f'backtest_portfolio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        portfolio_df.to_csv(portfolio_file, index=False)
        logger.info(f"Portfolio values saved to: {portfolio_file}")

        # Generate plot
        plot_file = os.path.join(log_dir, f'backtest_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        backtester.plot_results(plot_file)


def mode_optimize(config: dict):
    """
    Mode: Optimize strategy parameters

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("MODE: Parameter Optimization")
    logger.info("=" * 80)

    # Initialize components
    data_fetcher = DataFetcher(config)
    optimizer = StrategyOptimizer(config, data_fetcher)

    # Get parameters
    backtest_config = config.get('backtest', {})
    start_date = backtest_config.get('start_date', '2023-01-01')
    end_date = backtest_config.get('end_date', '2024-12-31')

    # Get tickers
    tickers_source = config.get('data', {}).get('tickers_source', 'nasdaq')
    tickers = data_fetcher.get_ticker_list(tickers_source)
    logger.info(f"Optimizing with {len(tickers)} stocks from {tickers_source}")

    # Run optimization
    opt_results = optimizer.optimize(start_date, end_date, tickers)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best {opt_results['metric_name']}: {opt_results['best_score']:.4f}")
    logger.info("\nBest Parameters:")

    best_params = opt_results['best_params']
    logger.info(f"  Gain Range: {best_params['gain_range']['min']:.1f}% - {best_params['gain_range']['max']:.1f}%")
    logger.info(f"  Volume Ratio (min): {best_params['volume_ratio']['min']:.2f}")
    logger.info(f"  Turnover Rate: {best_params['turnover_rate']['min']:.1f}% - {best_params['turnover_rate']['max']:.1f}%")
    logger.info(f"  Market Cap: ${best_params['market_cap']['min']:.0f}B - ${best_params['market_cap']['max']:.0f}B")
    logger.info("=" * 80)

    # Save results
    if config.get('logging', {}).get('save_results', True):
        log_dir = config.get('logging', {}).get('output_dir', 'logs')

        # Save optimization results
        results_file = os.path.join(log_dir, f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        optimizer.save_results(results_file)

        # Save best config
        best_config_file = os.path.join(log_dir, f'best_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        with open(best_config_file, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        logger.info(f"Best configuration saved to: {best_config_file}")

        # Analyze parameter importance
        importance_df = optimizer.analyze_parameter_importance()
        importance_file = os.path.join(log_dir, f'parameter_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        importance_df.to_csv(importance_file)
        logger.info(f"Parameter importance saved to: {importance_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Stock Selection System')
    parser.add_argument('--mode', type=str, choices=['select', 'backtest', 'optimize'],
                       default='select', help='Operation mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Stock Selection System Starting")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")

    try:
        # # Execute based on mode
        # if args.mode == 'select':
        #     mode_select_stocks(config)
        # elif args.mode == 'backtest':
        #     mode_backtest(config)
        # elif args.mode == 'optimize':
        #     mode_optimize(config)

        # mode_select_stocks(config)
        mode_backtest(config)

        logger.info("\nExecution completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
