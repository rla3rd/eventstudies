# Excel export

If needed, you can export your results to Excel directly using the `excel_exporter` module.

## Prerequisite

To use the Excel export functionalities, you must first install [`openpyxl`](https://openpyxl.readthedocs.io/) with the following command-line.

```bash
$ pip install openpyxl
```

**Note:** `openpyxl` is included as a dependency when installing the `eventstudies` package, so it should already be available.

Then add the following import statement at the beginning of your Python script:

```python
import eventstudies as es
from eventstudies import excel_exporter
```

The last line will attach to the `SingleEvent` and `MultipleEvents` classes a new function: `.to_excel()`.

## Usage

### Single Event Export

For a single event study, you can export results to Excel:

```python
from eventstudies import SingleEvent, excel_exporter
import numpy as np

# Create an event study
event = SingleEvent.MarketModel(
    ticker='AAPL',
    mkt_idx='SPY',
    event_date=np.datetime64('2007-01-09'),
    event_window=(-5, +10),
    est_size=252,
    buffer_size=21
)

# Export to Excel with default settings (Excel charts)
event.to_excel('results.xlsx')

# Export to Excel with chart as PNG image
event.to_excel('results.xlsx', chart_as_picture=True)
```

### Multiple Events Export

For multiple events, you can export the aggregate results:

```python
from eventstudies import MultipleEvents, excel_exporter

# Create multiple events study
events = MultipleEvents.from_list(
    event_list=[...],
    event_model=SingleEvent.MarketModel,
    event_window=(-5, +10)
)

# Export to Excel (includes summary sheet and individual event sheets)
events.to_excel('results.xlsx', event_details=True)

# Export without individual event sheets
events.to_excel('results.xlsx', event_details=False)

# Export with charts as PNG images
events.to_excel('results.xlsx', chart_as_picture=True)
```

## Parameters

### `to_excel()` method

- **`path`** (str): Path to save the Excel file
- **`chart_as_picture`** (bool, optional): If `True`, inserts charts as PNG images; if `False`, creates native Excel charts. Default: `False`
- **`event_details`** (bool, optional): For `MultipleEvents`, if `True`, creates individual sheets for each event. Default: `True`

## Excel File Structure

### Single Event Export

The Excel file contains a single sheet named "Summary" with:

1. **Specification Table**: Event details including:
   - Description
   - Event date
   - Event window (start and end)
   - Estimation size
   - Buffer size

2. **Results Table**: Statistical results including:
   - Event day number
   - AR (Abnormal Return)
   - Variance AR
   - CAR (Cumulative Abnormal Return)
   - Variance CAR
   - T-statistic
   - P-value

3. **Chart**: Visual representation of CAR and AR (either as Excel chart or PNG image)

### Multiple Events Export

The Excel file contains:

1. **Summary Sheet**: Aggregate results with:
   - Specification table
   - Aggregate results table (AAR, CAAR, etc.)
   - Chart showing CAAR

2. **Individual Event Sheets** (if `event_details=True`): One sheet per event with the same structure as single event export

## Alternative: Simple Pandas Export

You can also use pandas' built-in `to_excel()` method for a simple export of just the results table:

```python
# Simple export (just the results DataFrame)
event.results().to_excel('simple_results.xlsx')
```

This will export only the results table without charts or formatting.
