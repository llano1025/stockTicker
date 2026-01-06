"""
Stock selector implementing the 8-step filtering process
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


class StockSelector:
    """Implements 8-step stock selection strategy"""

    def __init__(self, config: dict, data_fetcher: DataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.selection_stats = {
            'step1_gainers': 0,
            'step2_volume_ratio': 0,
            'step3_turnover_rate': 0,
            'step4_market_cap': 0,
            'step5_volume_trend': 0,
            'step6_ma_alignment': 0,
            'step7_market_strength': 0,
            'step8_new_highs': 0,
            'final_selected': 0
        }

    def select_stocks(self, tickers: List[str], analysis_date: Optional[str] = None) -> List[Dict]:
        """
        Execute the 8-step stock selection process

        Args:
            tickers: List of ticker symbols to analyze
            analysis_date: Specific date for analysis (None for latest)

        Returns:
            List of selected stocks with details
        """
        logger.info("Starting 8-step stock selection process...")
        self._reset_stats()

        # Step 1: Filter by daily gains (3%-5%)
        step1_stocks = self._step1_filter_gainers(tickers)
        self.selection_stats['step1_gainers'] = len(step1_stocks)
        logger.info(f"Step 1: {len(step1_stocks)} stocks with 3%-8% gains")

        if not step1_stocks:
            logger.warning("No stocks passed Step 1 (gainers filter)")
            return []

        # Step 2: Remove volume ratio < 1
        step2_stocks = self._step2_filter_volume_ratio(step1_stocks)
        self.selection_stats['step2_volume_ratio'] = len(step2_stocks)
        logger.info(f"Step 2: {len(step2_stocks)} stocks with volume ratio >= 1")

        # Step 3: Remove turnover rate < 5% and > 10%
        step3_stocks = self._step3_filter_turnover_rate(step2_stocks)
        self.selection_stats['step3_turnover_rate'] = len(step3_stocks)
        logger.info(f"Step 3: {len(step3_stocks)} stocks with turnover rate 3%-10%")

        # Step 4: Remove market cap < $50B and > $200B
        step4_stocks = self._step4_filter_market_cap(step3_stocks)
        self.selection_stats['step4_market_cap'] = len(step4_stocks)
        logger.info(f"Step 4: {len(step4_stocks)} stocks with market cap $5B-$500B")

        # Step 5: Keep continuously increasing volume
        step5_stocks = self._step5_filter_volume_trend(step4_stocks)
        self.selection_stats['step5_volume_trend'] = len(step5_stocks)
        logger.info(f"Step 5: {len(step5_stocks)} stocks with increasing volume trend")

        # Step 6: Short-term MA aligns with 60-day line pointing upward
        step6_stocks = self._step6_filter_ma_alignment(step5_stocks)
        self.selection_stats['step6_ma_alignment'] = len(step6_stocks)
        logger.info(f"Step 6: {len(step6_stocks)} stocks with MA alignment")

        # Step 7: Stronger than the market
        step7_stocks = self._step7_filter_market_strength(step6_stocks)
        self.selection_stats['step7_market_strength'] = len(step7_stocks)
        logger.info(f"Step 7: {len(step7_stocks)} stocks stronger than market")

        # Step 8: Hit new highs at end of day
        step8_stocks = self._step8_filter_new_highs(step7_stocks)
        self.selection_stats['step8_new_highs'] = len(step8_stocks)
        logger.info(f"Step 8: {len(step8_stocks)} stocks hitting new highs")

        self.selection_stats['final_selected'] = len(step8_stocks)
        logger.info(f"Final selection: {len(step8_stocks)} stocks")

        return step8_stocks

    def _step1_filter_gainers(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Add stocks with 3%-5% gains to watchlist

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary of ticker -> DataFrame for stocks meeting criteria
        """
        min_gain = self.config['gain_range']['min']
        max_gain = self.config['gain_range']['max']

        return self.data_fetcher.get_gainers(tickers, min_gain, max_gain)

    def _step2_filter_volume_ratio(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 2: Remove stocks with volume ratio < 1

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        min_volume_ratio = self.config['volume_ratio']['min']
        filtered = {}

        for ticker, df in stocks.items():
            if 'Volume_Ratio' not in df.columns or df.empty:
                continue

            latest_volume_ratio = df['Volume_Ratio'].iloc[-1]

            if not np.isnan(latest_volume_ratio) and latest_volume_ratio >= min_volume_ratio:
                filtered[ticker] = df
                logger.debug(f"{ticker}: Volume ratio {latest_volume_ratio:.2f}")

        return filtered

    def _step3_filter_turnover_rate(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 3: Remove turnover rate < 5% and > 10%

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        min_turnover = self.config['turnover_rate']['min']
        max_turnover = self.config['turnover_rate']['max']
        filtered = {}

        for ticker, df in stocks.items():
            turnover_rate = self.data_fetcher.calculate_turnover_rate(ticker, df)

            if turnover_rate is None:
                continue

            if min_turnover <= turnover_rate <= max_turnover:
                filtered[ticker] = df
                logger.debug(f"{ticker}: Turnover rate {turnover_rate:.2f}%")

        return filtered

    def _step4_filter_market_cap(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 4: Remove market cap < $50B and > $200B

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        min_cap = self.config['market_cap']['min']
        max_cap = self.config['market_cap']['max']
        filtered = {}

        for ticker, df in stocks.items():
            info = self.data_fetcher.fetch_stock_info(ticker)

            if info is None:
                continue

            market_cap = info['market_cap']

            if min_cap <= market_cap <= max_cap:
                filtered[ticker] = df
                logger.debug(f"{ticker}: Market cap ${market_cap:.2f}B")

        return filtered

    def _step5_filter_volume_trend(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 5: Keep stocks with continuously increasing volume

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        periods = self.config['volume_trend']['periods']
        threshold = self.config['volume_trend']['threshold']
        filtered = {}

        for ticker, df in stocks.items():
            if len(df) < periods:
                continue

            # Check last N periods for volume increase
            recent_volumes = df['Volume'].iloc[-periods:].values
            increases = sum(1 for i in range(1, len(recent_volumes))
                          if recent_volumes[i] > recent_volumes[i-1])

            increase_ratio = increases / (periods - 1)

            if increase_ratio >= threshold:
                filtered[ticker] = df
                logger.debug(f"{ticker}: Volume increasing {increase_ratio:.2%} of time")

        return filtered

    def _step6_filter_ma_alignment(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 6: Select stocks where short-term MA aligns with 60-day line pointing upward

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        tolerance = self.config['moving_averages']['alignment_tolerance']
        filtered = {}

        for ticker, df in stocks.items():
            if 'MA5' not in df.columns or 'MA20' not in df.columns or 'MA60' not in df.columns:
                continue

            if len(df) < 60:
                continue

            latest_ma5 = df['MA5'].iloc[-1]
            latest_ma20 = df['MA20'].iloc[-1]
            latest_ma60 = df['MA60'].iloc[-1]
            prev_ma60 = df['MA60'].iloc[-2]

            # Check if MAs are aligned (MA5 > MA20 > MA60 or within tolerance)
            ma_aligned = (latest_ma5 >= latest_ma20 * (1 - tolerance) 
                          and
                         latest_ma20 >= latest_ma60 * (1 - tolerance)
                         )

            # Check if MA60 is pointing upward
            ma60_upward = latest_ma60 > prev_ma60

            if ma_aligned and ma60_upward:
                filtered[ticker] = df
                logger.debug(f"{ticker}: MA aligned and upward (MA5:{latest_ma5:.2f}, "
                           f"MA20:{latest_ma20:.2f}, MA60:{latest_ma60:.2f})")

        return filtered

    def _step7_filter_market_strength(self, stocks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 7: Keep stocks stronger than the market

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            Filtered dictionary
        """
        benchmark_ticker = self.config['market_strength']['benchmark']
        lookback_days = self.config['market_strength']['lookback_days']

        # Fetch market data
        market_df = self.data_fetcher.fetch_market_data(benchmark_ticker)

        if market_df is None or len(market_df) < lookback_days:
            logger.warning("Could not fetch market benchmark data, skipping Step 7")
            return stocks

        market_return = market_df['Returns'].iloc[-lookback_days:].sum()

        filtered = {}
        for ticker, df in stocks.items():
            if len(df) < lookback_days:
                continue

            stock_return = df['Returns'].iloc[-lookback_days:].sum()

            if stock_return > market_return:
                filtered[ticker] = df
                logger.debug(f"{ticker}: Stock return {stock_return:.2%} > "
                           f"Market return {market_return:.2%}")

        return filtered

    def _step8_filter_new_highs(self, stocks: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Step 8: Keep stocks that hit new highs and don't break below MA

        Args:
            stocks: Dictionary of ticker -> DataFrame

        Returns:
            List of dictionaries with stock details
        """
        lookback = self.config['new_high']['lookback_days']
        ma_period = self.config['new_high']['ma_support']

        selected_stocks = []

        for ticker, df in stocks.items():
            if len(df) < lookback:
                continue

            # Check if hitting new high
            is_new_high = df['Is_New_High'].iloc[-1] if 'Is_New_High' in df else False

            if not is_new_high:
                # Alternative check: current high >= recent period high
                recent_high = df['High'].iloc[-lookback:-1].max()
                current_high = df['High'].iloc[-1]
                is_new_high = current_high >= recent_high

            if not is_new_high:
                continue

            # Check if above MA support
            ma_col = f'MA{ma_period}'
            if ma_col not in df.columns:
                continue

            latest_close = df['Close'].iloc[-1]
            latest_ma = df[ma_col].iloc[-1]

            if latest_close >= latest_ma:
                stock_details = {
                    'ticker': ticker,
                    'close': latest_close,
                    'gain': df['Returns'].iloc[-1] * 100,
                    'volume_ratio': df['Volume_Ratio'].iloc[-1],
                    'ma5': df['MA5'].iloc[-1],
                    'ma20': df['MA20'].iloc[-1],
                    'ma60': df['MA60'].iloc[-1],
                    'is_new_high': True,
                    'above_ma_support': True,
                    'entry_signal': True
                }
                selected_stocks.append(stock_details)
                logger.debug(f"{ticker}: New high at ${latest_close:.2f}, above MA{ma_period}")

        return selected_stocks

    def get_selection_stats(self) -> Dict:
        """Get statistics on the selection process"""
        return self.selection_stats.copy()

    def _reset_stats(self):
        """Reset selection statistics"""
        for key in self.selection_stats:
            self.selection_stats[key] = 0
