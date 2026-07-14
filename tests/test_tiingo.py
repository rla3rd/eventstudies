# Copyright (C) 2023 Richard Albright
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
from eventstudies import tiingo


class TestTiingo(unittest.TestCase):
    def setUp(self):
        # Sample response data for Tiingo daily price endpoint
        self.mock_api_data = [
            {"date": "2023-01-03T00:00:00.000Z", "adjClose": 100.0, "close": 100.0},
            {"date": "2023-01-04T00:00:00.000Z", "adjClose": 105.0, "close": 105.0},
            {"date": "2023-01-05T00:00:00.000Z", "adjClose": 102.9, "close": 102.9},
        ]

    def test_download_prices_requires_key(self):
        # Ensure ValueError is raised if api_key is None and env variable is missing
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                tiingo.download_prices(["AAPL"], api_key=None)

    @patch("requests.Session.get")
    def test_download_prices_success(self, mock_get):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_data
        mock_get.return_value = mock_response

        # Execute download
        df = tiingo.download_prices(["AAPL"], api_key="test_key", delay=0.0)

        # Assertions
        mock_get.assert_called_once()
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ["date", "ticker", "adjClose"])
        self.assertTrue((df["ticker"] == "AAPL").all())
        self.assertEqual(df["adjClose"].iloc[0], 100.0)
        self.assertEqual(df["adjClose"].iloc[1], 105.0)

    @patch("requests.Session.get")
    def test_download_prices_multiple(self, mock_get):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_data
        mock_get.return_value = mock_response

        # Execute download with multiple tickers
        df = tiingo.download_prices(["AAPL", "MSFT"], api_key="test_key", delay=0.0)

        # Assertions
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(len(df), 6)
        self.assertEqual(set(df["ticker"].unique()), {"AAPL", "MSFT"})

    def test_to_logreturns(self):
        # Sample price data on actual trading days (Jan 3, Jan 4, Jan 5, 2023)
        prices = pd.DataFrame([
            {"date": pd.to_datetime("2023-01-03"), "ticker": "AAPL", "adjClose": 100.0},
            {"date": pd.to_datetime("2023-01-04"), "ticker": "AAPL", "adjClose": 105.0},
            {"date": pd.to_datetime("2023-01-05"), "ticker": "AAPL", "adjClose": 102.9},
            {"date": pd.to_datetime("2023-01-03"), "ticker": "MSFT", "adjClose": 200.0},
            {"date": pd.to_datetime("2023-01-04"), "ticker": "MSFT", "adjClose": 198.0},
            {"date": pd.to_datetime("2023-01-05"), "ticker": "MSFT", "adjClose": 202.0},
        ])

        # Execute transformation
        logreturns = tiingo.to_logreturns(prices, calendar="NYSE")

        # Assertions
        # Expecting rows for 2023-01-04 and 2023-01-05 (2023-01-03 will be dropped since t-1 is missing)
        self.assertEqual(len(logreturns), 2)
        self.assertEqual(list(logreturns.columns), ["AAPL", "MSFT"])
        
        # Verify index type and name
        self.assertEqual(logreturns.index.name, "date")
        self.assertTrue(isinstance(logreturns.index, pd.DatetimeIndex))

        # Check log returns values:
        # AAPL (Jan 4): ln(105/100) = 0.048790
        # MSFT (Jan 4): ln(198/200) = -0.010050
        np.testing.assert_almost_equal(logreturns.loc["2023-01-04", "AAPL"], np.log(105.0/100.0))
        np.testing.assert_almost_equal(logreturns.loc["2023-01-04", "MSFT"], np.log(198.0/200.0))


if __name__ == "__main__":
    unittest.main()
