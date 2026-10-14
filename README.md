# Hit Song Popularity Modeling

A statistical analysis of how audio features shape popularity in chart-topping songs, using Spotify audio features and Billboard Hot 100 data.

This repository contains the code and data for the paper:

**Ruoqing Huang, "Uncovering Audio Features Shaping Popularity in Chart-Topping Songs: A Statistical Approach,"** *American Journal of Student Research*, December 2024.
DOI: [10.70251/HYJR2348.24158174](https://doi.org/10.70251/HYJR2348.24158174)

## Overview

The project investigates which audio features (danceability, energy, valence, tempo, etc.) drive a song's chart performance:

- Spotify Web API data collection (track audio features, metadata)
- Billboard Hot 100 chart data (rank history)
- Statistical modeling: correlation analysis, PCA, clustering (K-Means, hierarchical), regression (linear, Lasso, Elastic Net, Random Forest, Gradient Boosting, XGBoost)
- Feature importance and dimensionality reduction to identify the strongest popularity predictors

## Repository Structure

- `src/main/python/` — Data collection and modeling code
  - `SpotifyJSONFetcher.py`, `SpotifyMetadataFetcher_Batch.py`, `SpotifyTrackSearch.py` — Spotify API data collection
  - `WikiBillboardFetcher.py` — Billboard chart data collection
  - `JSON2Column.py` — Flatten nested JSON to CSV columns
  - `CSVSplitAndZip.py` — Split large CSVs for processing
  - `data_analysis/` — Statistical analysis (correlation, PCA, clustering, regression, ensemble methods)
  - `MRA2/` — Multiple regression analysis scripts
- `data/` — Raw and processed datasets (CSVs via Git LFS)
- `results/` — Analysis outputs, figures, model results
- `docs/` — Research notes

## Requirements

- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

## License

MIT License — see [LICENSE](LICENSE).
