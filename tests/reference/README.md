# Reference and calibration tests (`stats-ref`)

Independent NumPy / SciPy / statsmodels implementations used to validate the
standard-library statistics in `openbench.stats`. Every module here calls
`pytest.importorskip`, so the core test run (without these libraries) skips them.

| Switch | Runs |
| --- | --- |
| (none) | Fast reference cross-checks |
| `OPENBENCH_RUN_SLOW=1` | Exact binary calibration (production and reference), n = 200 cases |
| `OPENBENCH_RUN_CALIBRATION=1` | Re-run of the frozen continuous calibration grid (about 15 minutes) |

```bash
python -m pip install -e ".[dev,config,metrics]" numpy scipy statsmodels
OPENBENCH_RUN_SLOW=1 python -m pytest tests/reference -v
```

`data/continuous_calibration_results.json` is the frozen protocol's recorded output
(design plan section 1.9.1.6); `data/ref_z120_fixture.json` is the n = 120 Monte Carlo
reference fixture.
