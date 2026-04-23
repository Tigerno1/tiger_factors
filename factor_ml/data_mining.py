from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import copy
import logging
import random
from typing import Any, Callable

from deap import base, creator, tools
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedFactor:
    name: str
    expression: str
    factor_type: str = "hybrid"
    complexity: str = "medium"
    source: str = ""


class DataPreprocessingService:
    """Preprocessing helpers for Tiger-native long panels.

    The native input/output schema is:
    - `date_`
    - `code`
    - OHLCV columns such as `open`, `high`, `low`, `close`, `volume`
    """

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        n_sigma: float = 3.0,
        method: str = "std",
    ) -> pd.Series:
        if column not in df.columns:
            raise ValueError(f"Column {column!r} does not exist in the frame")

        series = pd.to_numeric(df[column], errors="coerce")
        if method == "std":
            mean = series.mean()
            std = series.std(ddof=0)
            lower = mean - n_sigma * std
            upper = mean + n_sigma * std
            return (series < lower) | (series > upper)
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (series < lower) | (series > upper)
        raise ValueError(f"Unsupported outlier detection method: {method}")

    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "clip",
        n_sigma: float = 3.0,
        detection_method: str = "std",
    ) -> pd.DataFrame:
        frame = df.copy()
        outliers = self.detect_outliers(frame, column, n_sigma=n_sigma, method=detection_method)
        series = pd.to_numeric(frame[column], errors="coerce")

        if method == "clip":
            if detection_method == "std":
                center = series.mean()
                spread = series.std(ddof=0)
                lower = center - n_sigma * spread
                upper = center + n_sigma * spread
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
            frame.loc[series < lower, column] = lower
            frame.loc[series > upper, column] = upper
            return frame
        if method == "remove":
            return frame.loc[~outliers].copy()
        if method == "replace":
            frame.loc[outliers, column] = series.mean()
            return frame
        if method == "replace_median":
            frame.loc[outliers, column] = series.median()
            return frame
        raise ValueError(f"Unsupported outlier handling method: {method}")

    def incremental_update(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        on: str | list[str] = "date_",
        how: str = "outer",
    ) -> pd.DataFrame:
        left = existing_df.copy()
        right = new_df.copy()
        keys = [on] if isinstance(on, str) else list(on)

        for key in keys:
            if key in left.columns and "date" in key:
                left[key] = pd.to_datetime(left[key], errors="coerce")
            if key in right.columns and "date" in key:
                right[key] = pd.to_datetime(right[key], errors="coerce")

        if how not in {"outer", "inner"}:
            raise ValueError(f"Unsupported merge mode: {how}")

        combined = pd.concat([left, right], axis=0, ignore_index=True)
        subset = [key for key in keys if key in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
            sort_keys = [key for key in ("code", "date_") if key in combined.columns]
            if sort_keys:
                combined = combined.sort_values(sort_keys)
        else:
            combined = combined.drop_duplicates(keep="last")

        if how == "inner" and subset:
            common = None
            for key in subset:
                left_values = pd.Index(left[key].dropna().unique())
                right_values = pd.Index(right[key].dropna().unique())
                current = left_values.intersection(right_values)
                common = current if common is None else common.intersection(current)
            if common is not None and len(common) > 0 and len(subset) == 1:
                combined = combined[combined[subset[0]].isin(common)].copy()
        return combined.reset_index(drop=True)

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_columns: list[str] | None = None,
    ) -> tuple[bool, str]:
        if len(df) == 0:
            return False, "dataframe is empty"

        if required_columns:
            missing = sorted(set(required_columns) - set(df.columns))
            if missing:
                return False, f"missing required columns: {missing}"

        null_counts = df.isnull().sum()
        if int(null_counts.sum()) > 0:
            null_info = null_counts[null_counts > 0].to_dict()
            return False, f"contains nulls: {null_info}"

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                return False, f"column {col!r} contains infinite values"

        return True, "data quality check passed"

    def standardize_columns(
        self,
        df: pd.DataFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        frame = df.copy()
        default_mapping = {
            "日期": "date_",
            "date": "date_",
            "时间": "date_",
            "证券代码": "code",
            "代码": "code",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        mapping = column_mapping or default_mapping
        frame = frame.rename(columns=mapping)

        if "date_" in frame.columns:
            frame["date_"] = pd.to_datetime(frame["date_"], errors="coerce")
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str)
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        sort_keys = [key for key in ("code", "date_") if key in frame.columns]
        if not sort_keys:
            sort_keys = list(frame.columns[:1])
        return frame.sort_values(sort_keys).reset_index(drop=True)

    def fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
    ) -> pd.DataFrame:
        frame = df.copy()
        if method == "ffill":
            return frame.ffill()
        if method == "bfill":
            return frame.bfill()
        if method == "zero":
            return frame.fillna(0)
        if method == "interpolate":
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            frame[numeric_cols] = frame[numeric_cols].interpolate(limit_direction="both")
            return frame.ffill().bfill()
        if method in {"mean", "median"}:
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                series = frame[col]
                fill_value = series.mean() if method == "mean" else series.median()
                frame[col] = series.fillna(fill_value)
            return frame.ffill().bfill()
        raise ValueError(f"Unsupported fill method: {method}")


class FactorGeneratorService:
    """Generate, validate and pre-select factor expressions."""

    def __init__(self) -> None:
        self.operators = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "**": "pow",
            "%": "mod",
        }
        self.statistics = {
            "rank": "rank",
            "zscore": "zscore",
            "mean": "mean",
            "std": "std",
            "max": "max",
            "min": "min",
            "median": "median",
            "skew": "skew",
            "kurtosis": "kurtosis",
            "quantile": "quantile",
            "diff": "diff",
            "pct_change": "pct_change",
            "log": "log",
            "abs": "abs",
            "sqrt": "sqrt",
            "exp": "exp",
        }
        self.indicators = {
            "SMA": "simple moving average",
            "EMA": "exponential moving average",
            "RSI": "relative strength index",
            "MACD": "macd",
            "BBANDS": "bollinger bands",
            "STOCH": "stochastic",
            "ADX": "adx",
            "CCI": "cci",
            "ATR": "atr",
            "VOLATILITY": "volatility",
        }

    def generate_binary_combinations(
        self,
        base_factors: list[str],
        max_depth: int = 3,
        max_combinations: int = 100,
    ) -> list[str]:
        expressions: list[str] = []
        if len(base_factors) < 2:
            return expressions

        for left, right in combinations(base_factors, 2):
            for op in self.operators:
                expressions.append(f"({left} {op} {right})")

        if max_depth >= 2 and len(base_factors) >= 3:
            for _ in range(min(max_combinations // 2, 50)):
                a, b, c = random.sample(base_factors, 3)
                op1, op2 = random.sample(list(self.operators), 2)
                expressions.append(f"(({a} {op1} {b}) {op2} {c})")

        if max_depth >= 3 and len(base_factors) >= 4:
            for _ in range(min(max_combinations // 4, 25)):
                a, b, c, d = random.sample(base_factors, 4)
                op1, op2, op3 = random.sample(list(self.operators), 3)
                expressions.append(f"(({a} {op1} {b}) {op2} ({c} {op3} {d}))")

        return expressions[:max_combinations]

    def generate_statistical_combinations(
        self,
        base_factors: list[str],
        window_sizes: list[int] | None = None,
        max_combinations: int = 50,
    ) -> list[str]:
        window_sizes = window_sizes or [5, 10, 20, 60]
        expressions: list[str] = []
        for factor in base_factors:
            for stat_func in self.statistics:
                if stat_func in {"mean", "std", "max", "min", "median", "skew", "kurtosis"}:
                    for window in window_sizes:
                        expressions.append(f"({factor}.rolling({window}, min_periods=1).{stat_func}())")
                elif stat_func in {"diff", "pct_change", "abs"}:
                    expressions.append(f"({factor}.{stat_func}())")
                elif stat_func == "rank":
                    expressions.append(f"({factor}.rank(pct=True))")
                elif stat_func == "log":
                    expressions.append(f"np.log({factor})")
                elif stat_func == "sqrt":
                    expressions.append(f"np.sqrt({factor})")
                elif stat_func == "exp":
                    expressions.append(f"np.exp({factor})")
                elif stat_func == "zscore":
                    expressions.append(
                        f"(({factor} - {factor}.rolling(252, min_periods=1).mean()) / ({factor}.rolling(252, min_periods=1).std() + 1e-8))"
                    )
                elif stat_func == "quantile":
                    for q in [0.25, 0.5, 0.75]:
                        expressions.append(f"({factor}.rolling(252, min_periods=1).quantile({q}))")
        return expressions[:max_combinations]

    def generate_indicator_combinations(
        self,
        base_factors: list[str],
        price_column: str = "close",
        max_combinations: int = 30,
    ) -> list[str]:
        expressions: list[str] = []
        for factor in base_factors:
            for indicator in self.indicators:
                if indicator == "SMA":
                    for window in [5, 10, 20, 60]:
                        expressions.append(f"({factor} / SMA({price_column}, {window}))")
                        expressions.append(f"({factor} - SMA({price_column}, {window}))")
                elif indicator == "EMA":
                    for window in [5, 10, 20, 60]:
                        expressions.append(f"({factor} / EMA({price_column}, {window}))")
                elif indicator == "RSI":
                    expressions.append(f"({factor} * RSI({price_column}, 14))")
                elif indicator == "MACD":
                    expressions.append(f"({factor} * MACD({price_column}))")
        return expressions[:max_combinations]

    def generate_hybrid_factors(self, base_factors: list[str], n_factors: int = 100) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []

        binary = self.generate_binary_combinations(base_factors, max_combinations=int(n_factors * 0.4))
        factors.extend({"expression": expr, "type": "binary_operation", "complexity": "medium"} for expr in binary)

        statistical = self.generate_statistical_combinations(base_factors, max_combinations=int(n_factors * 0.3))
        factors.extend({"expression": expr, "type": "statistical", "complexity": "low"} for expr in statistical)

        indicator = self.generate_indicator_combinations(base_factors, max_combinations=int(n_factors * 0.2))
        factors.extend({"expression": expr, "type": "indicator_based", "complexity": "high"} for expr in indicator)

        while len(factors) < n_factors and len(base_factors) >= 2:
            left, right = random.sample(base_factors, 2)
            op = random.choice(list(self.operators))
            if random.random() < 0.3:
                stat_func = random.choice(list(self.statistics))
                if stat_func in {"mean", "std", "max", "min"}:
                    window = random.choice([5, 10, 20])
                    expr = f"{stat_func}({left} {op} {right}, {window})"
                else:
                    expr = f"{stat_func}({left} {op} {right})"
            else:
                expr = f"({left} {op} {right})"
            factors.append(
                {
                    "expression": expr,
                    "type": "random_hybrid",
                    "complexity": random.choice(["low", "medium", "high"]),
                }
            )

        random.shuffle(factors)
        return factors[:n_factors]

    def validate_expression(self, expression: str) -> tuple[bool, str]:
        if not expression or expression.strip() == "":
            return False, "expression is empty"
        if expression.count("(") != expression.count(")"):
            return False, "parentheses are unbalanced"
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()., _[]")
        for char in expression:
            if char not in allowed:
                return False, f"illegal character: {char}"
        has_operator = any(op in expression for op in self.operators)
        has_function = any(func in expression for func in self.statistics) or any(
            indicator in expression for indicator in self.indicators
        )
        if not (has_operator or has_function):
            return False, "expression has no operator or function"
        return True, ""

    def parse_expression(self, expression: str) -> dict[str, Any]:
        structure = {"expression": expression, "components": [], "operators": [], "functions": [], "depth": 0}
        for op in self.operators:
            if op in expression:
                structure["operators"].append(op)
        for func in self.statistics:
            if f"{func}(" in expression:
                structure["functions"].append(func)
        for func in self.indicators:
            if f"{func}(" in expression:
                structure["functions"].append(func)
        current_depth = 0
        max_depth = 0
        for char in expression:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1
        structure["depth"] = max_depth
        return structure

    def preselect_factors(
        self,
        factors: list[dict[str, Any]],
        factor_data_map: dict[str, pd.Series],
        return_data: pd.Series,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.5,
        min_valid_ratio: float = 0.7,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for factor_info in factors:
            expression = factor_info.get("expression", "")
            key = factor_info.get("name") or expression
            factor_values = factor_data_map.get(key)
            if factor_values is None:
                factor_values = factor_data_map.get(expression)
            if factor_values is None:
                continue
            aligned = pd.DataFrame({"factor": factor_values, "return": return_data}).dropna()
            if aligned.empty:
                continue
            valid_ratio = len(aligned) / len(return_data) if len(return_data) else 0.0
            ic = aligned["factor"].corr(aligned["return"])
            ir = ic / (aligned["factor"].std(ddof=0) + 1e-8) if pd.notna(ic) else 0.0
            if pd.isna(ic):
                continue
            if abs(ic) >= ic_threshold and abs(ir) >= ir_threshold and valid_ratio >= min_valid_ratio:
                enriched = dict(factor_info)
                enriched.update({"ic": float(ic), "ir": float(ir), "valid_ratio": float(valid_ratio)})
                selected.append(enriched)
        return selected


class GeneticFactorMiningService:
    """A compact DEAP-based genetic factor miner."""

    def __init__(
        self,
        base_factors: list[str],
        data: pd.DataFrame,
        return_column: str = "return",
        population_size: int = 50,
        n_generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        factor_calculator: Callable[[pd.DataFrame, str], pd.Series | pd.DataFrame | np.ndarray | list[float]] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.base = base
        self.creator = creator
        self.tools = tools
        self.base_factor_codes = list(base_factors)
        self.data = data.copy()
        self.return_column = return_column
        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.factor_calculator = factor_calculator
        self.random_state = random_state
        self.progress_callback: Callable[[int, int, float, float], None] | None = None

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.return_values = self.data[return_column] if return_column in self.data.columns else None
        self.base_factor_values: dict[str, dict[str, Any]] = {}
        self._precompute_base_factors()
        self._setup_genetic_algorithm()

    def set_progress_callback(self, callback: Callable[[int, int, float, float], None]) -> None:
        self.progress_callback = callback

    def _default_factor_calculator(self, data: pd.DataFrame, factor_code: str) -> pd.Series:
        if factor_code in data.columns:
            return pd.to_numeric(data[factor_code], errors="coerce")

        safe_env: dict[str, Any] = {column: pd.to_numeric(data[column], errors="coerce") for column in data.columns}
        safe_env.update(
            {
                "np": np,
                "pd": pd,
                "rank": lambda x: x.rank(pct=True),
                "zscore": lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8),
                "abs": np.abs,
                "sqrt": np.sqrt,
                "log": np.log,
                "exp": np.exp,
            }
        )
        result = eval(factor_code, {"__builtins__": {}}, safe_env)
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        if isinstance(result, (int, float, np.number)):
            return pd.Series([float(result)] * len(data), index=data.index)
        return pd.Series(result, index=data.index)

    def _precompute_base_factors(self) -> None:
        logger.info("precomputing %d base factors", len(self.base_factor_codes))
        for idx, factor_code in enumerate(self.base_factor_codes):
            try:
                if self.factor_calculator is None:
                    values = self._default_factor_calculator(self.data, factor_code)
                else:
                    try:
                        values = self.factor_calculator(self.data, factor_code)
                    except TypeError:
                        values = self.factor_calculator(factor_code)  # type: ignore[misc]

                if isinstance(values, pd.DataFrame):
                    values = values.iloc[:, 0]
                values = pd.Series(values, index=self.data.index, name=factor_code)
                values = pd.to_numeric(values, errors="coerce")
                if values.notna().sum() == 0:
                    logger.warning("base factor %s has no valid values", factor_code)
                    continue
                var_name = f"factor_{idx}"
                self.base_factor_values[var_name] = {"code": factor_code, "values": values}
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("failed to precompute %s: %s", factor_code, exc)

        if not self.base_factor_values:
            raise ValueError("No usable base factors were precomputed")

    def _setup_genetic_algorithm(self) -> None:
        if not hasattr(self.creator, "FitnessMax"):
            self.creator.create("FitnessMax", self.base.Fitness, weights=(1.0,))
        if not hasattr(self.creator, "Individual"):
            self.creator.create("Individual", list, fitness=self.creator.FitnessMax)

        self.toolbox = self.base.Toolbox()
        self.toolbox.register("individual", self._generate_random_individual)
        self.toolbox.register("population", self.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, indpb=0.2)
        self.toolbox.register("select", self.tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_factor)
        self.toolbox.register("clone", copy.deepcopy)
        self.stats = self.tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0.0)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def _generate_random_individual(self):
        var_names = list(self.base_factor_values.keys())
        individual = self.creator.Individual()
        if not var_names:
            individual.extend(["0.0"])
            return individual

        expr_type = random.choice(["single", "binary", "unary"])
        if expr_type == "single" or len(var_names) == 1:
            individual.extend([random.choice(var_names)])
            return individual

        if expr_type == "binary" and len(var_names) >= 2:
            left, right = random.sample(var_names, 2)
            op = random.choice(["+", "-", "*", "/"])
            individual.extend([f"({left} {op} {right})"])
            return individual

        var = random.choice(var_names)
        func = random.choice(["np.log", "np.sqrt", "np.abs", "rank"])
        if func == "rank":
            expr = f"rank({var})"
        else:
            expr = f"{func}({var})"
        individual.extend([expr])
        return individual

    def _compute_factor_expression(self, expr: str) -> pd.Series | None:
        try:
            safe_env: dict[str, Any] = {key: item["values"] for key, item in self.base_factor_values.items()}
            safe_env.update(
                {
                    "np": np,
                    "pd": pd,
                    "rank": lambda x: x.rank(pct=True),
                    "zscore": lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8),
                }
            )
            result = eval(expr, {"__builtins__": {}}, safe_env)
            if isinstance(result, (int, float, np.number)):
                return pd.Series([float(result)] * len(self.data), index=self.data.index)
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
            series = pd.Series(result, index=self.data.index)
            series = pd.to_numeric(series, errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan)
            if series.notna().sum() == 0:
                return None
            if series.notna().sum() < len(series) * 0.1:
                return None
            return series
        except Exception as exc:
            logger.debug("expression evaluation failed for %s: %s", expr, exc)
            return None

    def _evaluate_factor(self, individual: list[str]) -> tuple[float]:
        expr = individual[0]
        values = self._compute_factor_expression(expr)
        if values is None or values.notna().sum() < 10:
            return (0.0,)

        if self.return_values is not None:
            aligned = pd.DataFrame({"factor": values, "return": self.return_values}).dropna()
            if aligned.empty:
                return (0.0,)
            ic = aligned["factor"].corr(aligned["return"])
            score = 0.0 if pd.isna(ic) else float(abs(ic))
        else:
            score = float(values.std(ddof=0))
        return (score,)

    def _crossover(self, ind1, ind2):
        expr1 = ind1[0]
        expr2 = ind2[0]
        var_names = list(self.base_factor_values.keys())
        if len(var_names) < 1:
            return ind2, ind1

        vars1 = [var for var in var_names if var in expr1]
        vars2 = [var for var in var_names if var in expr2]
        if vars1 and vars2 and random.random() < 0.7:
            var1 = random.choice(vars1)
            var2 = random.choice(vars2)
            new_expr1 = expr1.replace(var1, var2, 1)
            new_expr2 = expr2.replace(var2, var1, 1)
            child1 = self.creator.Individual([new_expr1])
            child2 = self.creator.Individual([new_expr2])
            return child1, child2
        return ind2, ind1

    def _mutate(self, individual, indpb: float) -> tuple:
        expr = individual[0]
        var_names = list(self.base_factor_values.keys())
        operators = ["+", "-", "*", "/"]

        if random.random() < 0.3:
            for op in operators:
                if op in expr and random.random() < indpb:
                    expr = expr.replace(op, random.choice([item for item in operators if item != op]), 1)
                    break

        if random.random() < 0.3 and var_names:
            for var in var_names:
                if var in expr and random.random() < indpb:
                    other_vars = [item for item in var_names if item != var]
                    if other_vars:
                        expr = expr.replace(var, random.choice(other_vars), 1)
                        break

        if random.random() < 0.2 and var_names:
            var = random.choice([item for item in var_names if item in expr] or var_names)
            func = random.choice(["np.log", "np.sqrt", "np.abs"])
            expr = expr.replace(var, f"{func}({var})", 1)

        return (self.creator.Individual([expr]),)

    def run(self, generations: int | None = None) -> dict[str, Any]:
        ngen = generations or self.n_generations
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = self.tools.HallOfFame(maxsize=5)

        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)
        hall_of_fame.update(population)

        for generation in range(ngen):
            selected = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, selected))

            for left, right in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    child1, child2 = self.toolbox.mate(left, right)
                    left[0], right[0] = child1[0], child2[0]
                    del left.fitness.values
                    del right.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_prob:
                    mutated, = self.toolbox.mutate(mutant)
                    mutant[0] = mutated[0]
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for individual in invalid:
                individual.fitness.values = self.toolbox.evaluate(individual)

            population = offspring
            hall_of_fame.update(population)

            if self.progress_callback is not None:
                best = hall_of_fame[0] if len(hall_of_fame) > 0 else population[0]
                avg = float(np.mean([ind.fitness.values[0] for ind in population]))
                self.progress_callback(generation + 1, ngen, float(best.fitness.values[0]), avg)

        best = hall_of_fame[0] if len(hall_of_fame) > 0 else max(population, key=lambda ind: ind.fitness.values[0])
        return {
            "best_expression": best[0],
            "best_fitness": float(best.fitness.values[0]),
            "hall_of_fame": [ind[0] for ind in hall_of_fame],
            "population": [ind[0] for ind in population],
        }


__all__ = [
    "DataPreprocessingService",
    "FactorGeneratorService",
    "GeneticFactorMiningService",
    "GeneratedFactor",
]
