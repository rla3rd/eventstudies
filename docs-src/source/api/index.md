# API

The package implements two classes [`SingleEvent`](eventstudy.Single.md) and [`MultipleEvents`](eventstudy.Multiple.md)
to compute event studies respectively on single events (with measures such as Abnormal Returns (AR)
and Cumulative Abnormal Returns (CAR)) and on aggregates of events (with measures such as 
Average Abnormal Returns (AAR) and Cumulative Average Abnormal Returns (CAAR)).

The second class ([`MultipleEvents`](eventstudy.Multiple.md)) relies on the first one ([`SingleEvent`](eventstudy.Single.md)) 
as it basically performs a loop of single event studies and then aggregates them.

```eval_rst
.. toctree::
   :maxdepth: 2

   eventstudy.Single
   eventstudy.Multiple
   eventstudy.excelExporter
   util
```

