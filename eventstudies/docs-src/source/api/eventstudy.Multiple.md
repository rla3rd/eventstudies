```eval_rst

MultipleEvents Class
====================

.. automodule:: eventstudies.MultipleEvents

.. contents::
    :local:
    :depth: 1

Run the event study
-------------------

.. autosummary::
   :nosignatures:
   :toctree:

    eventstudies.MultipleEvents.__init__
    eventstudies.MultipleEvents.from_csv
    eventstudies.MultipleEvents.from_list
    eventstudies.MultipleEvents.from_text
    eventstudies.MultipleEvents.error_report

Import data
-----------

.. note:: Returns and factor data are directly imported at the single event study level.

.. autosummary::
   :nosignatures:
   :toctree:

    eventstudies.SingleEvent.import_FamaFrench
    eventstudies.SingleEvent.import_returns


Retrieve results
----------------

.. autosummary::
   :nosignatures:
   :toctree:

    eventstudies.MultipleEvents.plot
    eventstudies.MultipleEvents.results
    eventstudies.MultipleEvents.get_CAR_dist
    eventstudies.MultipleEvents.sign_test
    eventstudies.MultipleEvents.rank_test

```