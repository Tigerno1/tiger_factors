# Experimental Factor Algorithms

This package contains research-stage factor implementations that use
self-named data inputs and do not depend on a fixed external provider.

## Factors

### Market Breathing

Inputs:

- `date_`
- `code`
- `iv_surface_level`
- `iv_surface_skew`
- `iv_surface_curvature`
- `iv_term_slope`
- `option_spread`
- `option_volume`
- optional `option_open_interest`
- optional `realized_volatility`

Primary class:

- `MarketBreathingEngine`

### News Entropy

Inputs:

- `date_`
- `code`
- `event_topic`
- optional `event_sentiment`
- optional `event_novelty`
- optional `event_weight`

Primary class:

- `NewsEntropyEngine`

## Candidate Catalog

The package also includes a broader research catalog for future implementations.
Each entry records the name, category, required fields, optional fields, and a
short description.

Current catalog entries:

- `market_breathing`
- `vol_surface_torsion`
- `iv_realized_gap`
- `liquidity_pressure`
- `vol_of_vol_regime`
- `news_entropy`
- `event_density`
- `topic_dispersion`
- `news_novelty`
- `attention_spike`
- `information_pressure`
- `regime_anomaly`
- `panic_breath`

## Public helpers

- `experimental_factor_catalog()`
- `experimental_factor_spec(name)`
- `experimental_factor_names()`
- `market_breathing_factor_names()`
- `news_entropy_factor_names()`

## Notes

These engines are intentionally lightweight and research-oriented. They are
designed to be plugged into a future data layer or factor store workflow
without hard-coding any upstream provider.
