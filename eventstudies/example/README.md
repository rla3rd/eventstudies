# Example Directory

This directory contains example notebooks and data files demonstrating how to use the `eventstudies` package.

## Directory Structure

```
example/
├── data/              # Input data files (CSV format)
│   ├── 10K.csv        # Sample event data for multiple companies
│   ├── famafrench.csv # Fama-French factors data
│   ├── returns_GAFAM.csv  # Returns data for GAFA+M companies
│   ├── returns_small.csv  # Small sample returns data
│   └── sgen.csv       # Additional sample data
├── notebooks/         # Jupyter notebooks with examples
│   ├── example_notebook.ipynb  # Main example notebook (updated for current API)
│   ├── old_.ipynb     # Legacy notebook (old API)
│   ├── Untitled.ipynb # Untitled notebook
│   └── test.py        # Test script (old API)
└── outputs/           # Generated output files
    ├── export.xlsx    # Example Excel export
    ├── my_file.xlsx   # Additional output files
    └── my_file_mult.xlsx
```

## Getting Started

1. **Load the package and dependencies:**
   ```python
   import eventstudies as es
   from eventstudies import SingleEvent, MultipleEvents
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. **Import data:**
   ```python
   SingleEvent.import_returns('../data/returns_GAFAM.csv')
   SingleEvent.import_FamaFrench('../data/famafrench.csv')
   ```

3. **Run a single event study:**
   ```python
   event = SingleEvent.MarketModel(
       ticker='AAPL',
       mkt_idx='SPY',
       event_date=np.datetime64('2007-01-09'),
       event_window=(-2, 10),
       est_size=300,
       buffer_size=30
   )
   ```

4. **View results:**
   ```python
   print(event.results())
   event.plot()
   ```

## Data Files

- **10K.csv**: Contains event data with columns `ticker`, `mkt_idx`, and `event_date`
- **returns_GAFAM.csv**: Daily returns data for GAFA+M companies
- **famafrench.csv**: Fama-French factor data

## Notebooks

- **example_notebook.ipynb**: Main example notebook demonstrating:
  - Single event studies
  - Multiple event studies from CSV
  - Multiple event studies from Python lists
  - Plotting and results display

## Notes

- All paths in the notebooks are relative to the `notebooks/` directory (use `../data/` and `../outputs/`)
- Output files are saved to the `outputs/` directory
- The example notebook has been updated to use the current API (`SingleEvent`, `MultipleEvents`, etc.)
- The `10K.csv` file uses the current column names: `ticker`, `mkt_idx`, `event_date`

