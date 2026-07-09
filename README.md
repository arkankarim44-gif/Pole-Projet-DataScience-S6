# Graphical Models in Finance

Applying **Graphical Lasso (GLasso)** to estimate the sparse conditional dependency structure between assets of the **MSCI World Index**, using price and volume data retrieved from Yahoo Finance.

## Overview

This project builds a pipeline to:
1. Retrieve the list of MSCI World assets and their Yahoo Finance tickers.
2. Download and clean historical price/volume data.
3. Explore the dataset and standardize returns over configurable time ranges.
4. Estimate sparse precision (inverse covariance) matrices with Graphical Lasso to uncover the network of partial correlations between assets.

## Repository structure

```
.
├── data-retrieval.ipynb     # Fetch MSCI World tickers & download data (yfinance)
├── data-exploration.ipynb   # Load, inspect and visualize the asset dataset
├── glasso.ipynb             # Standardization + Graphical Lasso estimation
├── data/                    # Datasets (CSV) — see below
└── README.md
```

### Expected `data/` contents
- `msci-world-assets.csv` — raw list of MSCI World companies
- `msci-world-assets-tickers.csv` — cached list with matched Yahoo Finance tickers
- `yahoo-finance-dataset/` - one CSV per asset (~1160 assets)
- `cleaned_assets_price_volume_data.csv` - consolidated cleaned price/volume data
- `assets_default_mean_std_data.csv` - per-asset mean/std used for standardization

## Requirements

```bash
pip install yfinance pandas numpy matplotlib scikit-learn seaborn tqdm ipywidgets
```

## Usage

Run the notebooks in order:

1. **`data-retrieval.ipynb`** — set `use_cache = False` to refresh data from Yahoo Finance, or `use_cache = True` to reuse cached tickers.
2. **`data-exploration.ipynb`** — inspect and visualize the loaded assets.
3. **`glasso.ipynb`** — standardize over a chosen time window and fit the Graphical Lasso model.

## Method

Graphical Lasso estimates a sparse precision matrix Θ by solving:

$$\hat{\Theta} = \arg\max_{\Theta \succ 0} \; \log\det\Theta - \operatorname{tr}(S\Theta) - \lambda \lVert\Theta\rVert_1$$

where *S* is the empirical covariance and λ controls sparsity. Non-zero off-diagonal entries reveal the conditional (partial-correlation) dependencies between assets.

## License

MIT
