Results explorer
================

.. raw:: html

   <p>
     <a href="../_static/results-explorer.html" target="_blank" rel="noopener"
        style="display:inline-block;padding:10px 18px;background:#990F3D;
               color:#fff1e5;text-decoration:none;border-radius:2px;
               font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.04em;">
       Open the Results Explorer →
     </a>
   </p>

A self-contained HTML page for browsing a snapshot of ``torchgeo-bench``
benchmark results. The committed page is regenerated from
``results/models/*.csv``, ``results/profiles/*.csv``,
``results/intrinsic_dim/*.csv``, and the archived JSON snapshots with
:file:`experiments/scripts/regen_results_explorer.py`.

The page is regenerated automatically: the ``Results explorer`` GitHub Actions
workflow reruns the script and commits the refreshed page whenever a results
CSV lands on ``main``, so the explorer never lags behind ``results/``.  On a
pull request the same workflow regenerates the page and uploads it as a
build artifact for preview instead of requiring it to be committed.
