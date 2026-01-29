# Benchmarking Structural Estimators in Large Multi-Way Networks: PPML vs. Polyads

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Language](https://img.shields.io/badge/Language-Python%20%7C%20R-blue) ![Topic](https://img.shields.io/badge/Topic-Causal%20Inference-orange)

## 📌 Project Overview

This repository contains the code and results for the **Statistical Modeling Seminar** (ENSAE Paris, 2026).

The objective is to assess the performance of the **Polyad Estimator** proposed by [Resende, Lecué, Wilner, and Choné (2026)](https://arxiv.org/abs/2512.02203) against the industry standard **Pseudo-Poisson Maximum Likelihood (PPML)** in the context of high-dimensional fixed effects and sparse data.

While PPML is robust to heteroskedasticity, it suffers from the **Incidental Parameter Problem** and computational bottlenecks in large, sparse networks ($N \to \infty$). The Polyad estimator relies on a classification task approach to eliminate nuisance parameters (fixed effects) and theoretically provides unbiased estimates with valid confidence intervals in sparse settings.

## 🎯 Objectives

1.  **Data Engineering:** Construct a large-scale, sparse gravity dataset using **CEPII BACI** (HS-6 level) trade flows, injecting zero-flows to reflect real-world sparsity.
2.  **Implementation:** Deploy the Polyad estimator using the `polyads` library.
3.  **Benchmarking:** Compare estimates of Regional Trade Agreements (RTA) effects obtained via:
    * **PPML** (using `fixest` in R).
    * **Polyads** (using `polyads` in Python).
4.  **Inference:** Analyze the coverage of confidence intervals and bias under heavy sparsity.

## 📂 Data Source

We use international trade data from the **CEPII**:
* **BACI:** Bilateral trade flows at the HS-6 product level (High-Dimensional & Sparse).
* **Gravity Database:** Geodesic distances, GDPs, and RTA (Regional Trade Agreements) dummies.

> **Note:** Raw data files are not included in this repository due to size constraints. Please refer to the `data/README.md` for download instructions.

## 🛠️ Tech Stack

* **Python 3.10+**: Data processing & Polyad estimation.
    * *Key Libraries:* `polyads` (Resende et al.), `pandas`, `numpy`, `scikit-learn`.
* **R**: Benchmark estimation (PPML).
    * *Key Libraries:* `fixest` (Bergé), `data.table`.

## 🏗️ Repository Structure

```bash
├── data/               # Local data storage (ignored by git)
│   ├── raw/            # Original CEPII files
│   └── processed/      # Cleaned sparse matrices ready for regression
├── notebooks/          # Exploratory Data Analysis (EDA) & Prototyping
├── src/                # Source code
│   ├── preparation/    # Scripts to merge BACI + Gravity and inject zeros
│   └── estimation/     # Scripts running the Polyad estimator
├── results/            # Outputs: Tables, Coefficient Plots, Logs
└── requirements.txt    # Python dependencies