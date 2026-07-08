Ranking explorer
================

.. raw:: html

   <p>
     <a href="../_static/ranking_explorer.html" target="_blank" rel="noopener"
        style="display:inline-block;padding:10px 18px;background:#990F3D;
               color:#fff1e5;text-decoration:none;border-radius:2px;
               font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.04em;">
       Open the Ranking Explorer →
     </a>
   </p>

A methodology-aware, TabArena-style leaderboard of frozen geospatial foundation
models. One aggregation selector (average rank, ELO, improvability) re-ranks the
same backbones; four condition selectors (task, probe, bands, pooling) redefine
the score matrix underneath. A pairwise Kendall τ-b readout quantifies how much
the ranking reorders when a single condition flips.

The committed page is regenerated from ``results/all_results.csv`` with
:file:`experiments/scripts/regen_leaderboard.py`, which precomputes every
ranking slice with the ``evaluma`` package and inlines the result as JSON::

   python experiments/scripts/regen_leaderboard.py
