"""
Test script for Excel exporter functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import eventstudies as es
from eventstudies import SingleEvent, MultipleEvents
import numpy as np

# Import the excel exporter module (this attaches to_excel methods)
try:
    from eventstudies import excel_exporter
    print("✓ Successfully imported excel_exporter")
except ImportError as e:
    print(f"✗ Failed to import excel_exporter: {e}")
    sys.exit(1)

# Check if to_excel method is attached
if hasattr(SingleEvent, 'to_excel'):
    print("✓ SingleEvent.to_excel method is available")
else:
    print("✗ SingleEvent.to_excel method is NOT available")
    sys.exit(1)

if hasattr(MultipleEvents, 'to_excel'):
    print("✓ MultipleEvents.to_excel method is available")
else:
    print("✗ MultipleEvents.to_excel method is NOT available")
    sys.exit(1)

# Test with a simple event study
print("\n--- Testing Excel Export ---")

# Set up data paths
data_dir = project_root / "example" / "data"
returns_path = data_dir / "returns_GAFAM.csv"
famafrench_path = data_dir / "famafrench.csv"

if not returns_path.exists():
    print(f"✗ Returns file not found: {returns_path}")
    sys.exit(1)

if not famafrench_path.exists():
    print(f"✗ Fama-French file not found: {famafrench_path}")
    sys.exit(1)

# Import returns and Fama-French factors
try:
    SingleEvent.import_returns(path=str(returns_path))
    print("✓ Imported returns data")
except Exception as e:
    print(f"✗ Failed to import returns: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    SingleEvent.import_FamaFrench()
    print("✓ Imported Fama-French factors")
except Exception as e:
    print(f"✗ Failed to import Fama-French factors: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a test event
try:
    event = SingleEvent.MarketModel(
        ticker='AAPL',
        mkt_idx='SPY',
        event_date=np.datetime64('2007-01-09'),
        event_window=(-5, +10),
        est_size=252,
        buffer_size=21
    )
    print("✓ Created SingleEvent")
except Exception as e:
    print(f"✗ Failed to create SingleEvent: {e}")
    sys.exit(1)

# Test Excel export
output_dir = project_root / "example" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
excel_path = output_dir / "test_export.xlsx"

try:
    print(f"\nExporting to Excel: {excel_path}")
    event.to_excel(str(excel_path), chart_as_picture=False)
    print("✓ Successfully exported SingleEvent to Excel")
    
    if excel_path.exists():
        print(f"✓ Excel file created: {excel_path}")
        print(f"  File size: {excel_path.stat().st_size} bytes")
    else:
        print(f"✗ Excel file was not created")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to export to Excel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with chart as picture
excel_path_picture = output_dir / "test_export_picture.xlsx"
try:
    print(f"\nExporting to Excel with chart as picture: {excel_path_picture}")
    event.to_excel(str(excel_path_picture), chart_as_picture=True)
    print("✓ Successfully exported SingleEvent to Excel with picture chart")
    
    if excel_path_picture.exists():
        print(f"✓ Excel file created: {excel_path_picture}")
        print(f"  File size: {excel_path_picture.stat().st_size} bytes")
    else:
        print(f"✗ Excel file was not created")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to export to Excel with picture: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- All tests passed! ---")

