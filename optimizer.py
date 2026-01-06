"""
Optimizer for continuous improvement of stock selection strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import itertools
import logging
from copy import deepcopy
from data_fetcher import DataFetcher
from stock_selector import StockSelector
from backtester import Backtester

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimizer for finding best strategy parameters"""

    def __init__(self, base_config: dict, data_fetcher: DataFetcher):
        self.base_config = base_config
        self.data_fetcher = data_fetcher
        self.optimization_config = base_config.get('optimization', {})
        self.best_params = None
        self.best_score = float('-inf')
        self.optimization_results = []

    def optimize(self, start_date: str, end_date: str, tickers: List[str]) -> Dict:
        """
        Optimize strategy parameters

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: List of tickers to test

        Returns:
            Dictionary with best parameters and results
        """
        if not self.optimization_config.get('enabled', False):
            logger.info("Optimization disabled in config")
            return {'best_params': self.base_config, 'best_score': 0}

        method = self.optimization_config.get('method', 'grid_search')
        logger.info(f"Starting optimization using {method} method")

        if method == 'grid_search':
            return self._grid_search_optimize(start_date, end_date, tickers)
        elif method == 'random_search':
            return self._random_search_optimize(start_date, end_date, tickers)
        elif method == 'bayesian':
            return self._bayesian_optimize(start_date, end_date, tickers)
        else:
            logger.error(f"Unknown optimization method: {method}")
            return {'best_params': self.base_config, 'best_score': 0}

    def _grid_search_optimize(self, start_date: str, end_date: str,
                             tickers: List[str]) -> Dict:
        """
        Grid search optimization

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: List of tickers

        Returns:
            Optimization results
        """
        param_grid = self.optimization_config.get('parameters', {})
        metric_name = self.optimization_config.get('metrics', 'sharpe_ratio')

        logger.info(f"Grid search with {len(param_grid)} parameters")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        best_params = None
        best_score = float('-inf')
        results = []

        for i, combination in enumerate(param_combinations):
            # Create config with this parameter combination
            test_config = self._create_config_from_params(param_names, combination)

            # Run backtest
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}")
            score, metrics = self._evaluate_config(test_config, start_date, end_date, tickers)

            result = {
                'params': dict(zip(param_names, combination)),
                'score': score,
                'metrics': metrics
            }
            results.append(result)

            logger.info(f"Combination {i+1}: {metric_name} = {score:.4f}")

            # Update best
            if score > best_score:
                best_score = score
                best_params = test_config
                logger.info(f"New best score: {best_score:.4f}")

        self.best_params = best_params
        self.best_score = best_score
        self.optimization_results = results

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Optimization complete. Best {metric_name}: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'metric_name': metric_name
        }

    def _random_search_optimize(self, start_date: str, end_date: str,
                               tickers: List[str]) -> Dict:
        """
        Random search optimization

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: List of tickers

        Returns:
            Optimization results
        """
        param_grid = self.optimization_config.get('parameters', {})
        metric_name = self.optimization_config.get('metrics', 'sharpe_ratio')
        iterations = self.optimization_config.get('iterations', 50)

        logger.info(f"Random search with {iterations} iterations")

        best_params = None
        best_score = float('-inf')
        results = []

        for i in range(iterations):
            # Randomly sample parameters
            random_params = {}
            for param_name, param_values in param_grid.items():
                random_params[param_name] = np.random.choice(param_values)

            # Create config
            param_names = list(random_params.keys())
            param_values = list(random_params.values())
            test_config = self._create_config_from_params(param_names, param_values)

            # Run backtest
            logger.info(f"Testing iteration {i+1}/{iterations}")
            score, metrics = self._evaluate_config(test_config, start_date, end_date, tickers)

            result = {
                'params': random_params,
                'score': score,
                'metrics': metrics
            }
            results.append(result)

            logger.info(f"Iteration {i+1}: {metric_name} = {score:.4f}")

            # Update best
            if score > best_score:
                best_score = score
                best_params = test_config
                logger.info(f"New best score: {best_score:.4f}")

        self.best_params = best_params
        self.best_score = best_score
        self.optimization_results = results

        results.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Optimization complete. Best {metric_name}: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'metric_name': metric_name
        }

    def _bayesian_optimize(self, start_date: str, end_date: str,
                          tickers: List[str]) -> Dict:
        """
        Bayesian optimization (simplified version)

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: List of tickers

        Returns:
            Optimization results
        """
        logger.warning("Bayesian optimization not fully implemented, falling back to random search")
        return self._random_search_optimize(start_date, end_date, tickers)

    def _create_config_from_params(self, param_names: List[str],
                                   param_values: List[Any]) -> Dict:
        """
        Create configuration dictionary from parameter names and values

        Args:
            param_names: List of parameter names
            param_values: List of parameter values

        Returns:
            Configuration dictionary
        """
        config = deepcopy(self.base_config)

        for name, value in zip(param_names, param_values):
            # Parse parameter name (e.g., "gain_range_min" -> ["gain_range", "min"])
            parts = name.split('_')

            if len(parts) >= 2:
                # Handle nested config (e.g., gain_range.min)
                if parts[0] == 'gain' and parts[1] == 'range':
                    if parts[2] == 'min':
                        config['gain_range']['min'] = value
                    elif parts[2] == 'max':
                        config['gain_range']['max'] = value
                elif parts[0] == 'volume' and parts[1] == 'ratio':
                    if parts[2] == 'min':
                        config['volume_ratio']['min'] = value
                elif parts[0] == 'turnover' and parts[1] == 'rate':
                    if parts[2] == 'min':
                        config['turnover_rate']['min'] = value
                    elif parts[2] == 'max':
                        config['turnover_rate']['max'] = value
                elif parts[0] == 'market' and parts[1] == 'cap':
                    if parts[2] == 'min':
                        config['market_cap']['min'] = value
                    elif parts[2] == 'max':
                        config['market_cap']['max'] = value

        return config

    def _evaluate_config(self, config: Dict, start_date: str, end_date: str,
                        tickers: List[str]) -> Tuple[float, Dict]:
        """
        Evaluate a configuration by running backtest

        Args:
            config: Configuration to test
            start_date: Backtest start date
            end_date: Backtest end date
            tickers: List of tickers

        Returns:
            Tuple of (score, metrics dictionary)
        """
        metric_name = self.optimization_config.get('metrics', 'sharpe_ratio')

        try:
            # Create selector and backtester with this config
            selector = StockSelector(config, self.data_fetcher)
            backtester = Backtester(config, self.data_fetcher, selector)

            # Run backtest
            results = backtester.run_backtest(start_date, end_date, tickers)
            metrics = results['metrics']

            # Get score based on selected metric
            score = metrics.get(metric_name, 0)

            # Handle invalid scores
            if np.isnan(score) or np.isinf(score):
                score = float('-inf')

            return score, metrics

        except Exception as e:
            logger.error(f"Error evaluating config: {e}")
            return float('-inf'), {}

    def get_best_params(self) -> Dict:
        """Get the best parameters found"""
        return self.best_params

    def get_optimization_results(self) -> List[Dict]:
        """Get all optimization results"""
        return self.optimization_results

    def save_results(self, filepath: str):
        """
        Save optimization results to CSV

        Args:
            filepath: Path to save results
        """
        if not self.optimization_results:
            logger.warning("No optimization results to save")
            return

        # Convert results to DataFrame
        rows = []
        for result in self.optimization_results:
            row = result['params'].copy()
            row['score'] = result['score']
            row.update(result['metrics'])
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Optimization results saved to {filepath}")

    def analyze_parameter_importance(self) -> pd.DataFrame:
        """
        Analyze which parameters have the most impact on performance

        Returns:
            DataFrame with parameter importance analysis
        """
        if not self.optimization_results:
            logger.warning("No optimization results to analyze")
            return pd.DataFrame()

        # Extract all parameters and scores
        param_names = list(self.optimization_results[0]['params'].keys())
        param_importance = {}

        for param_name in param_names:
            # Group by parameter value and calculate average score
            param_values = {}

            for result in self.optimization_results:
                value = result['params'][param_name]
                score = result['score']

                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(score)

            # Calculate variance in average scores
            avg_scores = [np.mean(scores) for scores in param_values.values()]
            importance = np.std(avg_scores) if len(avg_scores) > 1 else 0

            param_importance[param_name] = {
                'importance': importance,
                'best_value': max(param_values.items(), key=lambda x: np.mean(x[1]))[0],
                'num_unique_values': len(param_values)
            }

        # Create DataFrame
        df = pd.DataFrame.from_dict(param_importance, orient='index')
        df = df.sort_values('importance', ascending=False)

        logger.info("Parameter importance analysis:")
        logger.info(f"\n{df}")

        return df
