"""
Event studies package for analyzing market events and their impact.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Rick Albright"
__description__ = "Events analyzer used to perform event study analysis"

from .single_event import SingleEvent
from .multiple_events import MultipleEvents

__all__ = ["SingleEvent", "MultipleEvents", "__version__"]
