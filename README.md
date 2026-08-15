# Victorian Electricity Demand Forecasting

Forecasting half-hourly electricity demand for a National Electricity Market
retailer, comparing two machine learning models against a seasonal-naive
baseline. Built for COSC2669 Case Studies in Data Science, RMIT University.

## Problem

An energy retailer buys electricity on the half-hourly wholesale spot market and
sells it on fixed tariffs. Forecast error translates directly into procurement
risk, so the models here are evaluated on dollars as well as megawatts.

## Data

Neither dataset is included in this repository; both are public at the links below.

- **AEMO Aggregated Price and Demand Data (VIC1)** — half-hourly settlement
  demand and regional reference price, January 2024 to December 2025.
  <https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data>
- **UCI Individual Household Electric Power Consumption** — one-minute readings
  from a single household near Paris, December 2006 to November 2010.
  <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

## Method

Both series are placed on a common half-hourly grid and given identical features:
autoregressive lags, rolling day and week statistics, calendar terms, and Fourier
terms for daily, weekly and annual cycles. Splits are chronological, holding out
the final 20 percent. Three models are compared: a seasonal-naive baseline,
histogram gradient boosting, and a multilayer perceptron.

Evaluation uses MAE, RMSE and MAPE, plus two operational metrics: peak-window MAE
and a dollar imbalance proxy weighting each error by the spot price prevailing in
that interval.

## Running it

    pip install pandas numpy scikit-learn matplotlib
    mkdir -p data/aemo data/uci
    # put the AEMO monthly CSVs in data/aemo
    # put household_power_consumption.txt in data/uci
    python demand_forecast.py

Outputs are written to `results/`.

## Results

Gradient boosting cut MAE by 86 percent against the baseline on the grid series,
from 536 MW to 76 MW. Its advantage over the MLP was 19.7 percent in megawatts
but 39.9 percent once errors were weighted by the prevailing spot price, since
the network's errors fell disproportionately in high-price intervals.

Both models performed far worse on the household series (MAPE 39.9 percent
against 1.8 percent for the grid), and the two were nearly indistinguishable
there. Individual occupancy is close to random at half-hourly resolution, while
the aggregate is predictable precisely because idiosyncratic behaviour cancels
out.

Error is not evenly distributed across the day. Both models are most accurate
overnight and least accurate between roughly 09:00 and 15:00, when rooftop solar
drives most of the variation in Victorian net demand. No weather features were
supplied, so this is error the current design cannot reach.
