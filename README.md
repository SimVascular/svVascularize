# svVascularize


[![Version](https://img.shields.io/pypi/v/svv.svg?logo=pypi&label=PyPI%20version)](https:://pypi.org/project/svv/)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20linux%20|%20windows-blue)
![Latest Release](https://img.shields.io/github/v/release/SimVascular/svVascularize?label=latest)
[![codecov](https://codecov.io/github/SimVascular/svVascularize/graph/badge.svg)](https://codecov.io/github/SimVascular/svVascularize)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15151168.svg)](https://doi.org/10.5281/zenodo.15151168)
[![Docs](https://img.shields.io/badge/docs-gh--pages-brightgreen)](https://simvascular.github.io/svVascularize/)
[![Telemetry](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/SimVascular/svVascularize/badge-data/telemetry.json)](https://sentry.io)
[![Basic smoke test](https://github.com/SimVascular/svVascularize/actions/workflows/basic-smoke-test.yml/badge.svg?branch=main&event=push)](https://github.com/SimVascular/svVascularize/actions/workflows/basic-smoke-test.yml?query=branch%3Amain)

<p align="left">
The svVascularize (svv) is an open-source API for automated vascular generation and multi-fidelity hemodynamic simulation
written in Python. Often small-caliber vessels are difficult or infeasible to obtain from experimental data sources 
despite playing important roles in blood flow regulation and cell microenvironments. svVascularize aims to provide tissue 
engineers and computational hemodynamic scientists with de novo vasculature that can easily be applied in 
biomanufacturing applications or computational fluid dynamic (CFD) analysis.
</p>

* **Website:** https://simvascular.github.io/svVascularize/
* **PyPi:** https://pypi.org/project/svv/
* **Source code:** https://github.com/SimVascular/svVascularize

## Installation

The package is published on PyPI as `svv`:

```bash
pip install svv
```

If you are installing into an existing environment and hit NumPy ABI errors such as
`_ARRAY_API not found` or `numpy.core.multiarray failed to import`, recreate the
environment with a NumPy version supported by your Python version. For Python
3.9–3.12, use NumPy 1.x:

```bash
python -m pip install --upgrade pip
python -m pip install --force-reinstall "numpy<2" svv
```

Python 3.13 requires NumPy 2.1 or newer:

```bash
python -m pip install --upgrade pip
python -m pip install --force-reinstall "numpy>=2.1" svv
```

On clusters / HPC systems (for example Stanford Sherlock), use a recent Python (3.9–3.13) and `pip`, and install into a
clean virtual environment or user site-packages.
