Based on the content of the paper "Advancing Asynchronous Multivariate Time Series Forecasting: Insights from Oceanographic Data," here is a draft for the `README.md` file for the repository containing the experiments from this work.

---

# AMTS Forecasting Experiments

This repository contains the code, data, and models used in the experiments from the paper **"Advancing Asynchronous Multivariate Time Series Forecasting: Insights from Oceanographic Data."** The paper focuses on forecasting tasks in Asynchronous Multivariate Time Series (AMTS), with experiments conducted on the newly introduced Port of Santos Oceanographic Data (PSOD) dataset.

## Introduction

Asynchronous Multivariate Time Series (AMTS) forecasting deals with time series data where variables have different sampling rates, missing values, and temporal misalignments. This repository supports the research that introduces AMTS forecasting techniques and evaluates various machine learning models on the Port of Santos Oceanographic Data (PSOD), which consists of five years of oceanographic measurements. The aim is to develop more flexible forecasting models capable of handling irregularities in time series data.

## Dataset

The dataset used for the experiments is the **Port of Santos Oceanographic Data (PSOD)**. This dataset consists of:

- Water current measurements
- Wave characteristics (height, speed, peak period)
- Wind data
- Sea surface height
- Numerical model data (SOFS)
- Astronomical tide data

The dataset exhibits irregular sampling intervals, missing data, and unaligned variables, making it a suitable benchmark for evaluating AMTS forecasting models.

## Models

We evaluate a range of machine learning models, including but not limited to:

- Chronos (Foundation model for time series forecasting)
- Continuous Graph Neural Networks (CGNN)
- Gap-Ahead (Graph-based model with time encoding)
- GRU (Gated Recurrent Unit)
- MoNODE (Modulated Neural ODEs)
- N-HITS (Neural Hierarchical Interpolation for Time Series)

Each model has been evaluated under varying levels of data loss (0%, 20%, 40%, 60%, and 80%) to benchmark their robustness in handling missing data.


## Results

The results of the experiments are measured using two key metrics:

- **Index of Agreement (IoA)**: Measures the agreement between forecasted and actual data values.
- **Mean Absolute Error (MAE)**: Evaluates the average magnitude of errors in the forecasted values.

We found that models with **time encoding** and architectures tailored for AMTS significantly outperform traditional models, especially as the ratio of missing data increases.

