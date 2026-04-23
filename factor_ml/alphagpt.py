from __future__ import annotations

import torch

from tiger_factors.factor_ml.alpha_constraints import FormulaConstraints, FormulaValidationResult, FormulaValidator
from tiger_factors.factor_evaluation import FactorEvaluation, evaluate_factor_panel
from tiger_factors.factor_ml.alpha_execution import AlphaDAGExecutor, CompiledFormula
from tiger_factors.factor_ml.alpha_ops import (
    LIQUIDITY_PANEL,
    PRICE_PANEL,
    RETURN_PANEL,
    RISK_PANEL,
    SCORE_PANEL,
    VOLUME_PANEL,
    OPS_CONFIG,
    AlphaOperator,
    TensorSpec,
    default_operator_registry,
    operator_name_list,
)


class AlphaFeatureEngineer:
    """Generic AlphaGPT-style feature engineer for multiple asset classes."""

    FEATURE_NAMES = [
        "RET",
        "LIQUIDITY",
        "PRESSURE",
        "VOLUME_ACCEL",
        "PRICE_DEVIATION",
        "LOG_VOLUME",
        "VOL_CLUSTER",
        "MOMENTUM_REVERSAL",
        "RELATIVE_STRENGTH",
        "RANGE_POSITION",
    ]
    INPUT_DIM = len(FEATURE_NAMES)
    FEATURE_SPECS = [
        RETURN_PANEL,
        LIQUIDITY_PANEL,
        SCORE_PANEL,
        VOLUME_PANEL,
        PRICE_PANEL,
        VOLUME_PANEL,
        RISK_PANEL,
        SCORE_PANEL,
        SCORE_PANEL,
        PRICE_PANEL,
    ]

    @staticmethod
    def _get(raw_dict: dict[str, torch.Tensor], primary: str, *aliases: str) -> torch.Tensor:
        for key in (primary, *aliases):
            if key in raw_dict:
                return raw_dict[key]
        raise KeyError(f"Missing required field: {primary}")

    @staticmethod
    def _get_optional(
        raw_dict: dict[str, torch.Tensor],
        primary: str,
        *aliases: str,
        like: torch.Tensor | None = None,
        fill: float = 0.0,
    ) -> torch.Tensor:
        for key in (primary, *aliases):
            if key in raw_dict:
                return raw_dict[key]
        if like is None:
            raise KeyError(f"Missing optional field: {primary}")
        return torch.full_like(like, fill)

    @staticmethod
    def _robust_norm(t: torch.Tensor) -> torch.Tensor:
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)

    @staticmethod
    def _liquidity_health(liquidity: torch.Tensor, fdv: torch.Tensor) -> torch.Tensor:
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def _liquidity_proxy(volume: torch.Tensor, dollar_volume: torch.Tensor) -> torch.Tensor:
        ratio = dollar_volume / (torch.abs(volume) + 1.0)
        return torch.tanh(torch.log1p(ratio))

    @staticmethod
    def compute_features(raw_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        c = AlphaFeatureEngineer._get(raw_dict, "close", "adj_close", "price_close")
        o = AlphaFeatureEngineer._get(raw_dict, "open", "price_open")
        h = AlphaFeatureEngineer._get(raw_dict, "high", "price_high")
        l = AlphaFeatureEngineer._get(raw_dict, "low", "price_low")
        v = AlphaFeatureEngineer._get(raw_dict, "volume", "shares_volume", "contracts_volume")
        liq = AlphaFeatureEngineer._get_optional(raw_dict, "liquidity", "adv", "dollar_volume", like=c, fill=0.0)
        fdv = AlphaFeatureEngineer._get_optional(raw_dict, "fdv", "market_cap", "float_market_cap", like=c, fill=0.0)

        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = torch.where(
            fdv.abs() > 1e-9,
            AlphaFeatureEngineer._liquidity_health(liq, fdv),
            AlphaFeatureEngineer._liquidity_proxy(v, liq),
        )
        range_hl = h - l + 1e-9
        pressure = torch.tanh(((c - o) / range_hl) * 3.0)
        vol_prev = torch.roll(v, 1, dims=1)
        vol_chg = (v - vol_prev) / (vol_prev + 1.0)
        fomo = torch.clamp(vol_chg - torch.roll(vol_chg, 1, dims=1), -5.0, 5.0)

        pad20 = torch.zeros((c.shape[0], 19), device=c.device, dtype=c.dtype)
        c_pad = torch.cat([pad20, c], dim=1)
        ma20 = c_pad.unfold(1, 20, 1).mean(dim=-1)
        dev = (c - ma20) / (ma20 + 1e-9)
        log_vol = torch.log1p(v)

        ret_sq = ret ** 2
        pad10 = torch.zeros((ret_sq.shape[0], 9), device=c.device, dtype=c.dtype)
        ret_sq_pad = torch.cat([pad10, ret_sq], dim=1)
        vol_cluster = torch.sqrt(ret_sq_pad.unfold(1, 10, 1).mean(dim=-1) + 1e-9)

        pad5 = torch.zeros((ret.shape[0], 4), device=c.device, dtype=c.dtype)
        ret_pad = torch.cat([pad5, ret], dim=1)
        mom = ret_pad.unfold(1, 5, 1).sum(dim=-1)
        momentum_rev = (mom * torch.roll(mom, 1, dims=1) < 0).float()

        delta = c - torch.roll(c, 1, dims=1)
        gains = torch.relu(delta)
        losses = torch.relu(-delta)
        pad14 = torch.zeros((gains.shape[0], 13), device=c.device, dtype=c.dtype)
        gains_pad = torch.cat([pad14, gains], dim=1)
        losses_pad = torch.cat([pad14, losses], dim=1)
        avg_gain = gains_pad.unfold(1, 14, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, 14, 1).mean(dim=-1)
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rel_strength = (100 - (100 / (1 + rs)) - 50) / 50

        close_pos = (c - l) / (h - l + 1e-9)

        return torch.stack(
            [
                AlphaFeatureEngineer._robust_norm(ret),
                liq_score,
                pressure,
                AlphaFeatureEngineer._robust_norm(fomo),
                AlphaFeatureEngineer._robust_norm(dev),
                AlphaFeatureEngineer._robust_norm(log_vol),
                AlphaFeatureEngineer._robust_norm(vol_cluster),
                momentum_rev,
                AlphaFeatureEngineer._robust_norm(rel_strength),
                close_pos,
            ],
            dim=1,
        )


class AlphaFormulaVM:
    """Compatibility wrapper over the DAG compiler/executor."""

    def __init__(self, feature_dim: int = AlphaFeatureEngineer.INPUT_DIM, constraints: FormulaConstraints | None = None):
        self.feature_dim = feature_dim
        self.executor = AlphaDAGExecutor(
            feature_dim,
            constraints=constraints,
            feature_specs=AlphaFeatureEngineer.FEATURE_SPECS[:feature_dim],
        )
        self.feature_specs: list[TensorSpec] = AlphaFeatureEngineer.FEATURE_SPECS[:feature_dim]
        self.operators: tuple[AlphaOperator, ...] = default_operator_registry()
        self.validator = FormulaValidator(feature_dim, constraints=constraints, feature_specs=self.feature_specs)

    def validate(self, formula_tokens: list[int]) -> FormulaValidationResult:
        return self.validator.validate(formula_tokens)

    def compile(self, formula_tokens: list[int]) -> CompiledFormula:
        return self.executor.compile(formula_tokens)

    def execute(self, formula_tokens: list[int] | CompiledFormula, feat_tensor: torch.Tensor) -> torch.Tensor | None:
        try:
            return self.executor.evaluate(formula_tokens, feat_tensor)
        except Exception:
            return None

    def execute_many(
        self,
        formulas: list[list[int] | CompiledFormula],
        feat_tensor: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        try:
            return self.executor.evaluate_many(formulas, feat_tensor)
        except Exception:
            return None


__all__ = [
    "AlphaFeatureEngineer",
    "AlphaFormulaVM",
    "AlphaDAGExecutor",
    "AlphaOperator",
    "CompiledFormula",
    "FactorEvaluation",
    "FormulaConstraints",
    "FormulaValidationResult",
    "FormulaValidator",
    "OPS_CONFIG",
    "TensorSpec",
    "default_operator_registry",
    "evaluate_factor_panel",
    "operator_name_list",
]
