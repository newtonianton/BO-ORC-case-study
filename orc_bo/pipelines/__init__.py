"""Optimization pipelines.

* :mod:`orc_bo.pipelines.onestage` - one-stage BO directly in one-hot fluid space.
* :mod:`orc_bo.pipelines.twostage` - two-stage property-targeting then realization.

Both support a ``mode`` of ``"pure"`` (snap to one-hot vertices) or ``"mixture"``
(snap to binary-mixture edges), and share the same fluid-loading, candidate-realization
and result-recording helpers in :mod:`orc_bo.pipelines.common`.
"""
from __future__ import annotations
