# Benchmarking Structural Gravity Estimators: PPML vs. Polyads on Sparse Trade Data

[![Institution](https://img.shields.io/badge/Institution-ENSAE%20Paris-003366)](https://www.ensae.fr)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)](https://www.r-project.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

## Overview

This project benchmarks two structural gravity estimators—the industry-standard **Pseudo-Poisson Maximum Likelihood (PPML)** and the novel **Polyad estimator** (Resende, Lecué, Wilner & Choné, 2026)—on large-scale, sparse international trade data. The work was produced for the *Statistical Modeling Seminar* at **ENSAE Paris (2026)**.

PPML is the standard workhorse of the trade gravity literature, valued for its robustness to heteroskedasticity and consistent handling of zero trade flows. However, it is subject to the **Incidental Parameter Problem (IPP)** and faces computational bottlenecks as network dimensionality grows. The Polyad estimator reformulates fixed-effects elimination as a **classification task**, offering theoretically unbiased estimates with valid confidence intervals under heavy sparsity—a setting where PPML is known to break down.

## Research Question

> Does the Polyad estimator deliver less-biased and better-calibrated estimates of Regional Trade Agreement (RTA) effects than PPML when applied to high-dimensional, sparse gravity networks?

## Data

| Source | Description |
|---|---|
| **CEPII BACI** | Bilateral trade flows at the HS-6 product level — provides a naturally high-dimensional, sparse bilateral trade matrix |
| **CEPII Gravity Database** | Country-pair covariates: geodesic distances, GDPs, contiguity, common language, and RTA membership dummies |

Zero-flows are injected systematically into the BACI dataset to amplify sparsity and stress-test both estimators under conditions representative of real-world disaggregated trade networks.

> Raw data files are not included due to size constraints. See `data/README.md` for download and placement instructions from the CEPII website.

## Methodology

1. **Data Engineering** — Merge BACI trade flows with gravity covariates at the HS-6 level and inject zero-flows to construct a sparse bilateral trade matrix suitable for benchmarking.
2. **PPML Benchmark** — Estimate RTA effects using the `fixest` package in R, absorbing high-dimensional exporter × year and importer × year fixed effects via the Frisch–Waugh–Lovell theorem.
3. **Polyad Estimation** — Deploy the `polyads` Python library (Resende et al., 2026) to eliminate fixed effects via a classification sub-problem and recover structural parameters under sparsity.
4. **Evaluation** — Compare point estimates, standard errors, and confidence interval coverage across both estimators at varying sparsity levels.

**Stack:** Python 3.10+ (`polyads`, `pandas`, `numpy`, `scikit-learn`) · R (`fixest`, `data.table`)

## Key Results

| Estimator | RTA Coefficient | 95% CI Coverage | Notes |
|---|---|---|---|
| PPML (`fixest`) | — | — | High-dimensional FE benchmark |
| Polyads | — | — | Classification-based FE elimination |

Full coefficient tables and coverage plots will be available in `results/` upon completion of the benchmarking pipeline.

## Replication

```bash
# 1. Clone the repository
git clone https://github.com/VarnelT/estimator-Polyads-vs-PPML.git
cd estimator-Polyads-vs-PPML

# 2. Download CEPII BACI and Gravity data, place in data/raw/
#    Instructions: see data/README.md

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Build the sparse gravity dataset
python src/preparation/merge_and_inject_zeros.py

# 5. Run Polyads estimation (Python)
python src/estimation/run_polyads.py

# 6. Run PPML benchmark (R)
Rscript src/estimation/run_ppml.R
```

Results are written to `results/` as CSV tables and PNG coefficient plots.

## References

- Resende, G., Lecué, G., Wilner, L., & Choné, P. (2026). *Polyad Estimator for Large Multi-Way Networks*. arXiv:2512.02203.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The Log of Gravity. *The Review of Economics and Statistics*, 88(4), 641–658.
- Bergé, L. (2018). Efficient estimation of maximum likelihood models with multiple fixed-effects. *The Stata Journal*, 18(4), 796–820.

---

*Statistical Modeling Seminar — ENSAE Paris, 2026*
