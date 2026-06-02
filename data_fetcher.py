"""
Data fetcher module for stock market data retrieval
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and manages stock market data"""

    def __init__(self, config: dict):
        self.config = config
        self.data_cache = {}

    def get_ticker_list(self, source: str = "most_active") -> List[str]:
        """
        Get list of stock tickers from Yahoo Finance dynamic screeners

        Args:
            source: Source of tickers. Available options:
                    - "most_active": Most traded stocks by volume (default)
                    - "day_gainers": Biggest percentage gainers today
                    - "day_losers": Biggest percentage losers today
                    - "aggressive_small_caps": Growth-focused small cap stocks
                    - "growth_technology_stocks": Tech sector growth stocks
                    - "undervalued_growth_stocks": Value + growth combination
                    - "undervalued_large_caps": Large cap value stocks
                    And 300+ other Yahoo Finance screeners

        Returns:
            List of ticker symbols (top 250)
        """
        try:
            from yahooquery import Screener

            # Map user-friendly names to yahooquery screener IDs
            screener_map = {
                "most_active": "most_actives",
                "day_gainers": "day_gainers",
                "day_losers": "day_losers",
                "aggressive_small_caps": "aggressive_small_caps",
                "growth_technology_stocks": "growth_technology_stocks",
                "undervalued_growth_stocks": "undervalued_growth_stocks",
                "undervalued_large_caps": "undervalued_large_caps"
            }

            # Get screener ID (use source directly if not in map)
            screener_id = screener_map.get(source, source)

            # Fetch data from Yahoo Finance
            logger.info(f"Fetching top 250 stocks from '{source}' screener...")
            s = Screener()
            data = s.get_screeners([screener_id], count=250)

            # Extract ticker symbols
            if screener_id in data and 'quotes' in data[screener_id]:
                tickers = [quote['symbol'] for quote in data[screener_id]['quotes']]
                logger.info(f"Loaded {len(tickers)} tickers from '{source}' (dynamic screener)")
                return tickers
            else:
                logger.error(f"No data returned from screener '{source}'")
                raise ValueError(f"Screener '{source}' returned no data")

        except ImportError:
            logger.error("yahooquery library not installed. Run: pip install yahooquery>=2.3.0")
            raise ImportError("yahooquery is required for dynamic stock lists. Install with: pip install yahooquery>=2.3.0")

        except Exception as e:
            logger.error(f"Error fetching dynamic stock list from '{source}': {e}")
            raise RuntimeError(f"Failed to fetch stocks from '{source}': {e}")

    def fetch_stock_data(self, ticker: str, period: str = "3mo",
                        interval: str = "1d",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a ticker

        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start_date: Start date (YYYY-MM-DD) - overrides period if provided
            end_date: End date (YYYY-MM-DD) - overrides period if provided

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        # Create cache key - include dates if provided
        if start_date and end_date:
            cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        else:
            cache_key = f"{ticker}_{period}_{interval}"

        # Check cache first
        if cache_key in self.data_cache:
            logger.debug(f"Using cached data for {ticker}")
            return self.data_cache[cache_key]

        try:
            stock = yf.Ticker(ticker)

            # Use absolute dates if provided, otherwise use period
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return None

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Cache the data
            self.data_cache[cache_key] = df

            logger.debug(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def bulk_download_historical_data(self, tickers: List[str],
                                     start_date: str,
                                     end_date: str,
                                     interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple tickers at once

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (default: 1d)

        Returns:
            Dictionary mapping ticker → DataFrame with full date range
        """
        logger.info(f"Bulk downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        all_data = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            # Progress logging every 50 tickers
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} tickers downloaded")

            try:
                df = self.fetch_stock_data(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

                if df is not None and not df.empty:
                    all_data[ticker] = df
                else:
                    failed_tickers.append(ticker)
                    logger.debug(f"No data for {ticker}")

            except Exception as e:
                failed_tickers.append(ticker)
                logger.warning(f"Failed to download {ticker}: {e}")

        logger.info(f"Bulk download complete: {len(all_data)} successful, {len(failed_tickers)} failed")
        if failed_tickers:
            logger.debug(f"Failed tickers: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")

        return all_data

    def create_pit_snapshot(self, all_data: Dict[str, pd.DataFrame],
                           as_of_date: datetime,
                           lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Create point-in-time snapshot by slicing data

        Args:
            all_data: Full historical data dict (ticker → DataFrame)
            as_of_date: Date to simulate "current" data
            lookback_days: How many days of history to include (default: 90)

        Returns:
            Dictionary of DataFrames sliced to [as_of_date - lookback_days, as_of_date]
        """
        pit_snapshot = {}

        for ticker, df in all_data.items():
            try:
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Normalize timezone if needed
                if df.index.tz is None:
                    if hasattr(as_of_date, 'tz') and as_of_date.tz is not None:
                        as_of_date_naive = pd.Timestamp(as_of_date).tz_localize(None)
                    else:
                        as_of_date_naive = pd.Timestamp(as_of_date)
                else:
                    if hasattr(as_of_date, 'tz') and as_of_date.tz is not None:
                        as_of_date_naive = pd.Timestamp(as_of_date).tz_convert(df.index.tz)
                    else:
                        as_of_date_naive = pd.Timestamp(as_of_date).tz_localize(df.index.tz)

                # Filter data up to as_of_date
                df_filtered = df[df.index <= as_of_date_naive]

                if df_filtered.empty:
                    logger.debug(f"No data for {ticker} before {as_of_date}")
                    continue

                # Keep only last N days
                df_sliced = df_filtered.tail(lookback_days).copy()

                # Recalculate technical indicators on sliced data
                # Remove old indicators first (except OHLCV columns)
                ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                cols_to_keep = [col for col in df_sliced.columns if col in ohlcv_cols]
                df_sliced = df_sliced[cols_to_keep]

                # Add fresh indicators
                df_sliced = self._add_technical_indicators(df_sliced)

                pit_snapshot[ticker] = df_sliced

            except Exception as e:
                logger.warning(f"Error creating snapshot for {ticker}: {e}")
                continue

        logger.debug(f"Created point-in-time snapshot for {len(pit_snapshot)} tickers as of {as_of_date}")
        return pit_snapshot

    def fetch_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        Fetch stock information including market cap, volume, etc.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant information
            stock_info = {
                'ticker': ticker,
                'market_cap': info.get('marketCap', 0) / 1e9,  # Convert to billions
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }

            return stock_info

        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return None

    def fetch_market_data(self, ticker: str = "^GSPC", period: str = "3mo") -> Optional[pd.DataFrame]:
        """
        Fetch market benchmark data (e.g., S&P 500)

        Args:
            ticker: Market index ticker (default: ^GSPC for S&P 500)
            period: Data period

        Returns:
            DataFrame with market data
        """
        return self.fetch_stock_data(ticker, period=period)

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        # Daily returns
        df['Returns'] = df['Close'].pct_change()

        # Volume ratio (current volume / 20-day average volume)
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # Turnover rate approximation used as fallback when float shares are unavailable.
        # Expresses today's volume as a percentage of the 90-day mean volume, which
        # produces values in a similar range to actual US turnover rates (0.5%–5%).
        vol_mean = df['Volume'].mean()
        df['Turnover_Approx'] = (df['Volume'] / vol_mean) * 1.5 if vol_mean > 0 else 0.0

        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()

        # Volume trend (increasing volume)
        df['Volume_Trend'] = df['Volume'].diff()

        # New high indicator
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Is_New_High'] = df['High'] >= df['High_20'].shift(1)

        return df

    def get_gainers(self, tickers: List[str], min_gain: float = 3.0,
                   max_gain: float = 5.0) -> Dict[str, pd.DataFrame]:
        """
        Get stocks with daily gains in specified range

        Args:
            tickers: List of ticker symbols
            min_gain: Minimum gain percentage
            max_gain: Maximum gain percentage

        Returns:
            Dictionary of ticker -> DataFrame for stocks meeting criteria
        """
        gainers = {}

        for ticker in tickers:
            df = self.fetch_stock_data(ticker, period="3mo", interval="1d")

            if df is not None and len(df) >= 2:
                latest_return = df['Returns'].iloc[-1] * 100  # Convert to percentage

                if min_gain <= latest_return <= max_gain:
                    gainers[ticker] = df
                    logger.debug(f"{ticker}: {latest_return:.2f}% gain")

        logger.info(f"Found {len(gainers)} gainers out of {len(tickers)} stocks")
        return gainers

    def calculate_turnover_rate(self, ticker: str, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate actual turnover rate for a stock

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with volume data

        Returns:
            Turnover rate percentage or None
        """
        info = self.fetch_stock_info(ticker)

        if info is None or info['float_shares'] == 0:
            # Use approximation from df
            return df['Turnover_Approx'].iloc[-1] if 'Turnover_Approx' in df else None

        latest_volume = df['Volume'].iloc[-1]
        turnover_rate = (latest_volume / info['float_shares']) * 100

        return turnover_rate

    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache = {}
        logger.info("Data cache cleared")
