# Copyright (C) 2020 Jean-Baptiste Lemaire
# Copyright (C) 2023 Richard Albright
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Event studies package for analyzing market events and their impact.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Richard Albright"
__description__ = "Events analyzer used to perform event study analysis"

from .single_event import SingleEvent
from .multiple_events import MultipleEvents
from . import tiingo

__all__ = ["SingleEvent", "MultipleEvents", "tiingo", "__version__"]
