# FlowFM
# Flow-Based Probabilistic Forecasting for Financial Returns



- `FinGAN(best-by-val)` as the baseline
- `FlowFMPlus_Aux` as the best flow-based model obtained so far



## Files in this repository

- `FinGAN.py`  
  Main FinGAN implementation, including:
  - data splitting
  - return construction
  - generator / discriminator definitions
  - training loops for different loss combinations
  - evaluation functions

- `DataCleaning.py`  
  Preprocessing helper for constructing per-ticker CSV files from the raw stock and ETF datasets.

- `FinGAN-example.py`  
  Example usage script for the FinGAN codebase.

- `run_full_paper.py`  
  Script for running the FinGAN baseline across the paper universe and producing per-ticker results.

- `flow_adapter_aux.py`  
  Conditional flow-matching model and adapter used to make the flow model compatible with the FinGAN evaluation pipeline.

- `run_flowfm_aux_31.py`  
  Script for training and evaluating `FlowFMPlus_Aux` on the 31-ticker universe.

- `stocks-etfs-list.csv`  
  Mapping file used to match stocks to their sector ETFs for excess-return construction.

- `Results/`  
  Result files generated from completed experiments.

---

## Data setup

This repository assumes the presence of per-ticker CSV files inside a `data/` folder.

Expected data files include:
- stock-level CSVs such as `AMZN.csv`, `HD.csv`, etc.
- ETF CSVs such as `XLY.csv`, `XLP.csv`, etc.

These ticker CSVs are generated from the original raw files using `DataCleaning.py`.

If you only have the raw source files, you will also need:
- `ETFs-data.csv`
- `Stocks-data.csv`

These raw data files are not included in this repository.

---


