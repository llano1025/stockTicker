"""
Backtesting engine for stock selection strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging
from data_fetcher import DataFetcher
from stock_selector import StockSelector

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for evaluating stock selection strategy"""

    def __init__(self, config: dict, data_fetcher: DataFetcher, selector: StockSelector):
        self.config = config
        self.data_fetcher = data_fetcher
        self.selector = selector
        self.backtest_config = config.get('backtest', {})

        # Trading parameters
        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.position_size = self.backtest_config.get('position_size', 0.1)
        self.max_positions = self.backtest_config.get('max_positions', 10)
        self.hold_days = self.backtest_config.get('hold_days', 5)
        self.stop_loss = self.backtest_config.get('stop_loss', -0.05)
        self.take_profit = self.backtest_config.get('take_profit', 0.10)

        # Results
        self.trades = []
        self.portfolio_values = []
        self.metrics = {}

    def run_backtest(self, start_date: str, end_date: str, tickers: List[str]) -> Dict:
        """
        Run backtest over specified date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: List of tickers to consider

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Initialize
        self.trades = []
        self.portfolio_values = []
        current_capital = self.initial_capital
        open_positions = []

        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate trading dates (we'll simulate weekly selection)
        current_date = start
        trading_days = []

        while current_date <= end:
            trading_days.append(current_date)
            current_date += timedelta(days=7)  # Weekly selection

        logger.info(f"Generated {len(trading_days)} trading periods")

        # Main backtest loop
        for i, selection_date in enumerate(trading_days):
            logger.debug(f"Processing {selection_date.strftime('%Y-%m-%d')} ({i+1}/{len(trading_days)})")

            # Close expired positions
            open_positions, realized_pnl = self._close_expired_positions(
                open_positions, selection_date, current_capital
            )
            current_capital += realized_pnl

            # Check stop loss and take profit
            open_positions, sl_tp_pnl = self._check_stop_loss_take_profit(
                open_positions, selection_date, current_capital
            )
            current_capital += sl_tp_pnl

            # Select new stocks if we have capital and open slots
            if len(open_positions) < self.max_positions:
                # For backtesting, we need historical data at this point
                # This is a simplified approach - in reality, you'd need point-in-time data
                try:
                    selected_stocks = self.selector.select_stocks(tickers)

                    # Open new positions
                    new_positions = self._open_positions(
                        selected_stocks, current_capital, len(open_positions), selection_date
                    )

                    if new_positions:
                        capital_used = sum(pos['entry_value'] for pos in new_positions)
                        current_capital -= capital_used
                        open_positions.extend(new_positions)
                        logger.debug(f"Opened {len(new_positions)} new positions, "
                                   f"used ${capital_used:,.2f}")

                except Exception as e:
                    logger.warning(f"Error selecting stocks on {selection_date}: {e}")

            # Calculate portfolio value
            portfolio_value = current_capital + self._calculate_open_positions_value(
                open_positions, selection_date
            )
            self.portfolio_values.append({
                'date': selection_date,
                'value': portfolio_value,
                'cash': current_capital,
                'positions': len(open_positions)
            })

            logger.debug(f"Portfolio value: ${portfolio_value:,.2f}, "
                        f"Cash: ${current_capital:,.2f}, "
                        f"Positions: {len(open_positions)}")

        # Close all remaining positions
        for position in open_positions:
            self._close_position(position, end, current_capital)

        # Calculate metrics
        self.metrics = self._calculate_metrics()

        logger.info("Backtest completed")
        logger.info(f"Total trades: {len(self.trades)}")
        logger.info(f"Final portfolio value: ${self.portfolio_values[-1]['value']:,.2f}")
        logger.info(f"Total return: {self.metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win rate: {self.metrics['win_rate']:.2%}")

        return {
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'metrics': self.metrics
        }

    def _open_positions(self, selected_stocks: List[Dict], current_capital: float,
                       current_positions: int, entry_date: datetime) -> List[Dict]:
        """
        Open new positions based on selected stocks

        Args:
            selected_stocks: List of selected stock details
            current_capital: Available capital
            current_positions: Number of currently open positions
            entry_date: Position entry date

        Returns:
            List of new position dictionaries
        """
        positions = []
        available_slots = self.max_positions - current_positions
        position_value = current_capital * self.position_size

        for stock in selected_stocks[:available_slots]:
            if current_capital < position_value:
                break

            ticker = stock['ticker']
            entry_price = stock['close']
            shares = int(position_value / entry_price)

            if shares == 0:
                continue

            position = {
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'shares': shares,
                'entry_value': shares * entry_price,
                'hold_days': 0,
                'status': 'open'
            }

            positions.append(position)

        return positions

    def _close_position(self, position: Dict, exit_date: datetime,
                       current_capital: float) -> float:
        """
        Close a position and record the trade

        Args:
            position: Position dictionary
            exit_date: Exit date
            current_capital: Current capital

        Returns:
            Realized P&L
        """
        # Fetch exit price (using close price on exit date)
        df = self.data_fetcher.fetch_stock_data(position['ticker'], period="1y")

        if df is None or len(df) == 0:
            exit_price = position['entry_price']  # Assume no change
            logger.warning(f"Could not fetch exit price for {position['ticker']}, "
                         f"using entry price")
        else:
            # Find closest date
            df.index = pd.to_datetime(df.index)
            # Handle timezone awareness
            exit_date_tz = self._normalize_timezone(exit_date, df.index)
            closest_date = df.index[df.index <= exit_date_tz][-1] if any(df.index <= exit_date_tz) else df.index[0]
            exit_price = df.loc[closest_date, 'Close']

        exit_value = position['shares'] * exit_price
        pnl = exit_value - position['entry_value']
        pnl_pct = (exit_price / position['entry_price'] - 1) * 100

        trade = {
            'ticker': position['ticker'],
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_days': (exit_date - position['entry_date']).days,
            'exit_reason': position.get('exit_reason', 'time_limit')
        }

        self.trades.append(trade)
        position['status'] = 'closed'

        return pnl

    def _close_expired_positions(self, positions: List[Dict], current_date: datetime,
                                 current_capital: float) -> Tuple[List[Dict], float]:
        """
        Close positions that have reached hold period

        Args:
            positions: List of open positions
            current_date: Current date
            current_capital: Current capital

        Returns:
            Tuple of (remaining positions, realized P&L)
        """
        remaining_positions = []
        total_pnl = 0

        for position in positions:
            days_held = (current_date - position['entry_date']).days

            if days_held >= self.hold_days:
                position['exit_reason'] = 'time_limit'
                pnl = self._close_position(position, current_date, current_capital)
                total_pnl += pnl
            else:
                position['hold_days'] = days_held
                remaining_positions.append(position)

        return remaining_positions, total_pnl

    def _check_stop_loss_take_profit(self, positions: List[Dict], current_date: datetime,
                                     current_capital: float) -> Tuple[List[Dict], float]:
        """
        Check and execute stop loss and take profit orders

        Args:
            positions: List of open positions
            current_date: Current date
            current_capital: Current capital

        Returns:
            Tuple of (remaining positions, realized P&L)
        """
        remaining_positions = []
        total_pnl = 0

        for position in positions:
            # Fetch current price
            df = self.data_fetcher.fetch_stock_data(position['ticker'], period="1mo")

            if df is None or len(df) == 0:
                remaining_positions.append(position)
                continue

            df.index = pd.to_datetime(df.index)
            # Handle timezone awareness
            current_date_tz = self._normalize_timezone(current_date, df.index)
            closest_date = df.index[df.index <= current_date_tz][-1] if any(df.index <= current_date_tz) else df.index[0]
            current_price = df.loc[closest_date, 'Close']

            # Calculate return
            position_return = (current_price / position['entry_price']) - 1

            # Check stop loss
            if position_return <= self.stop_loss:
                position['exit_reason'] = 'stop_loss'
                pnl = self._close_position(position, current_date, current_capital)
                total_pnl += pnl
                logger.debug(f"Stop loss triggered for {position['ticker']} at {position_return:.2%}")

            # Check take profit
            elif position_return >= self.take_profit:
                position['exit_reason'] = 'take_profit'
                pnl = self._close_position(position, current_date, current_capital)
                total_pnl += pnl
                logger.debug(f"Take profit triggered for {position['ticker']} at {position_return:.2%}")

            else:
                remaining_positions.append(position)

        return remaining_positions, total_pnl

    def _normalize_timezone(self, dt: datetime, df_index: pd.DatetimeIndex) -> datetime:
        """
        Normalize datetime to match DataFrame index timezone

        Args:
            dt: Datetime object to normalize
            df_index: DataFrame DatetimeIndex to match timezone with

        Returns:
            Timezone-aware datetime matching the DataFrame index timezone
        """
        if df_index.tz is None:
            # DataFrame is timezone-naive, ensure dt is also naive
            if hasattr(dt, 'tz') and dt.tz is not None:
                return dt.tz_localize(None)
            return dt
        else:
            # DataFrame is timezone-aware
            if hasattr(dt, 'tz') and dt.tz is not None:
                # dt is already timezone-aware, convert to df_index timezone
                return dt.tz_convert(df_index.tz)
            else:
                # dt is timezone-naive, localize to df_index timezone
                return pd.Timestamp(dt).tz_localize(df_index.tz)

    def _calculate_open_positions_value(self, positions: List[Dict],
                                        current_date: datetime) -> float:
        """
        Calculate current value of open positions

        Args:
            positions: List of open positions
            current_date: Current date

        Returns:
            Total value of open positions
        """
        total_value = 0

        for position in positions:
            df = self.data_fetcher.fetch_stock_data(position['ticker'], period="1mo")

            if df is None or len(df) == 0:
                current_price = position['entry_price']
            else:
                df.index = pd.to_datetime(df.index)
                # Handle timezone awareness
                current_date_tz = self._normalize_timezone(current_date, df.index)
                closest_date = df.index[df.index <= current_date_tz][-1] if any(df.index <= current_date_tz) else df.index[0]
                current_price = df.loc[closest_date, 'Close']

            total_value += position['shares'] * current_price

        return total_value

    def _calculate_metrics(self) -> Dict:
        """
        Calculate backtest performance metrics

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades or not self.portfolio_values:
            return {}

        # Total return
        final_value = self.portfolio_values[-1]['value']
        total_return = (final_value / self.initial_capital) - 1

        # Win rate
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0

        # Average win/loss
        wins = [trade['pnl_pct'] for trade in self.trades if trade['pnl'] > 0]
        losses = [trade['pnl_pct'] for trade in self.trades if trade['pnl'] < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        total_losses = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate returns for Sharpe ratio
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['returns'] = portfolio_df['value'].pct_change()

        # Sharpe ratio (assuming 252 trading days per year)
        returns_mean = portfolio_df['returns'].mean()
        returns_std = portfolio_df['returns'].std()
        sharpe_ratio = (returns_mean / returns_std) * np.sqrt(52) if returns_std > 0 else 0  # Weekly data

        # Maximum drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min()

        metrics = {
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'losing_trades': len(self.trades) - winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'total_pnl': final_value - self.initial_capital
        }

        return metrics

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get backtest results as DataFrame

        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio values as DataFrame

        Returns:
            DataFrame with portfolio values over time
        """
        if not self.portfolio_values:
            return pd.DataFrame()

        return pd.DataFrame(self.portfolio_values)

    def plot_results(self, save_path: str = None):
        """
        Plot backtest results

        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt

            portfolio_df = self.get_portfolio_dataframe()

            if portfolio_df.empty:
                logger.warning("No data to plot")
                return

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Portfolio value over time
            axes[0].plot(portfolio_df['date'], portfolio_df['value'], label='Portfolio Value')
            axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Value ($)')
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].legend()
            axes[0].grid(True)

            # Drawdown
            portfolio_df['cummax'] = portfolio_df['value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax']
            axes[1].fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, alpha=0.3, color='red')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Drawdown')
            axes[1].set_title('Drawdown Over Time')
            axes[1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
