# 🏠 Istanbul Housing Market Analysis & Price Prediction

<p align="center">
  <img src="docs/assets/istanbul_banner.jpg" width="900" alt="Istanbul skyline">
</p>


A full data science pipeline on a real Istanbul apartment listings dataset (24,767 listings, 38 districts) — from raw data cleaning through exploratory analysis to a tuned, evaluated regression model.

## Project Highlights

- **Rigorous data cleaning methodology**: every quality issue documented and flagged, never silently dropped (Notebook 01).
- **21-question exploratory analysis** uncovering real market patterns — and real analysis pitfalls, like a raw-price district ranking that turned out to be confounded by apartment size (see [Key Findings](#key-findings)).
- **A caught-and-fixed data leak in the cleaning logic itself**: residual analysis on the trained model (Notebook 04) surfaced 16 listings with implausibly low prices that had slipped past the original error flag — fixed by looping back to Notebook 01.
- **Final model**: Random Forest + target encoding, MedAE ≈ 1.1M TRY (~16% of median price) on log-transformed price, selected after confirming that raw R²/RMSE on price is unstable due to a handful of ultra-luxury outliers.

## Data Source
This project uses the ["Istanbul Apartment Prices 2026"](https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026) dataset, published on Kaggle.

## Repository Structure

```
istanbul-housing-analysis/
├── data/
│   ├── raw/                                  # original scraped dataset (untouched, never modified)
│   └── processed/
│       └── istanbul_apartments_flagged.csv   # cleaned + flagged dataset (output of Notebook 01)
├── notebooks/
│   ├── 01_data_cleaning.ipynb                # data quality assessment & flagging
│   ├── 02_EDA.ipynb                          # 21-question exploratory analysis
│   ├── 03_ML.ipynb                           # first preprocessing + baseline regression pipeline
│   ├── 04_Model_Improvement.ipynb            # cross-validation, tuning, encoding comparison, residual analysis
│   └── 05_Final_Model_Report.ipynb           # final model, regression plot, project summary, try-it-yourself predictor
├── requirements.txt
└── README.md
```

## Methodology

1. **Data Cleaning (Notebook 01)** — checked for missing values, duplicates, quasi-duplicates, and logical/physical inconsistencies (impossible room counts, bathroom counts, floor counts, building ages, and prices — both too high *and* too low). Every issue is tracked in its own `flag_*` column; **no row is ever deleted** in this notebook.
2. **Exploratory Data Analysis (Notebook 02)** — answered 21 questions on price drivers (size, rooms, age, floor, location, market segment) with mean/median comparisons, correlation checks (with and without flagged rows — flagged rows alone were shown to distort correlations by up to 8x), and a formal IQR-based statistical outlier pass cross-referenced against the rule-based flags.
3. **Modeling (Notebooks 03-05)** — `log1p(price)` as the target (price is strongly right-skewed), `price_per_sqm` explicitly excluded as a feature (it's derived from the target — leakage), one-hot vs. target encoding compared, Linear Regression / Random Forest / Gradient Boosting compared with cross-validation, hyperparameter tuning, and a full residual analysis.

## Key Findings

- Price is heavily right-skewed (skewness ≈ 63) — median (6.75M TRY) is far more representative than the mean (13.4M TRY).
- Apartment size (`gross_sqm`, r≈0.57) and `bathroom_count` (r≈0.49) are the strongest single-variable price correlates.
- A naive "which district is most expensive" ranking by raw price is misleading — it's confounded by district-level differences in typical apartment size. Price per m² is the more reliable comparison metric.
- Location does matter, but size still edges it out as the dominant price driver once both compete inside the same model.
- Rare, statistically legitimate luxury outliers make RMSE/R² on raw price unstable; Median Absolute Error is used as the primary real-world accuracy metric.

## Model Performance (Final Model — Random Forest, target-encoded)

| Metric | Value |
|---|---|
| R² (log-price, 3-fold CV) | 0.839 ± 0.004 |
| Median Absolute Error | ~1.10M TRY (~16% of median price) |

**Known limitation:** the model is not reliable for ultra-luxury properties (>~50M TRY) — too few training examples in that range. Key missing features (sea view, exact renovation quality) likely explain the remaining error in high-variance districts like Beşiktaş and Şişli.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Bokeh

## Running This Project

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_cleaning.ipynb
```

Run the notebooks in order (01 → 05); each one reads the previous notebook's output from `data/processed/`.

## Author

Built by Abdelhafidh — freelance Data & Software Engineer.

## License

MIT
