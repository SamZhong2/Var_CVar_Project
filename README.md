# Mutual Fund VaR, CVaR, and Sector Stress Testing Dashboard

A Streamlit dashboard for analyzing the risk of an equal-weight mutual fund portfolio using:

- historical $\tau$-horizon returns
- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- bootstrap simulation
- sector-based stress testing
- sensitivity analysis over `d`, $\tau$, and $\delta$

The app is designed around an equity mutual fund dataset and a matching time-varying sector exposure dataset.

## Features

- Select a portfolio of `d` mutual funds
- Choose:
  - start date `ts`
  - end date `te`
  - confidence level $\alpha$
  - horizon $\tau$
  - rolling step $\delta$
- Compute:
  - historical VaR and CVaR
  - stressed VaR and CVaR
  - bootstrap VaR and CVaR
- Apply sector shock scenarios
- Explore sensitivity analysis:
  - VaR/CVaR vs number of funds `d`
  - VaR/CVaR vs horizon $\tau$
  - VaR/CVaR vs rolling step $\delta$
  - VaR heatmap across $\tau × \delta$


## Data requirements

The app expects these CSV files in a `data/` folder:

### 1. `us_equity_adj_close.csv`

Wide format:

- one column named `as_of`
- one column for each fund ID

Example:

| as_of | B13779 | B14678 | ... |
|------|--------|--------|-----|
| 1999-12-31 | 20.2885 | NaN | ... |

### 2. `us_equity_sectors.csv`

Panel format:

- one column named `as_of`
- one column named `ask_id`
- sector exposure columns ending in `_pct_net`

Example:

| as_of | ask_id | equity_econ_sector_technology_pct_net | ... |
|------|--------|----------------------------------------|-----|
| 2017-01 | B00056 | 12.34 | ... |

The app automatically:
- converts `ask_id` to the fund identifier
- uses the latest available sector exposure on or before the selected start date
- converts sector percentages into decimal weights

## Installation

Clone the repository:

```bash
git clone https://github.com/SamZhong2/Var_CVar_Project.git
cd Var_CVar_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## How to use the app

1. Put the required CSV files in the `data/` folder.
2. Launch the app with `streamlit run app.py`.
3. In the sidebar:
   - set the price data path
   - set the sector data path
   - choose `ts`, `te`, `d`, $\tau$, $\delta$, and $\alpha$
   - choose a fund-selection mode
   - set sector shocks
4. Click the run button.
5. Review:
   - selected funds
   - base VaR/CVaR
   - stressed VaR/CVaR
   - bootstrap risk metrics
   - sensitivity analysis plots

## Methodology summary

### Rolling horizon returns

For fund `i`, the $\tau$-horizon return is:

$r^{(i)}_{t,\tau} = \frac{p^{(i)}_{t+\tau} - p^{(i)}_t}{p^{(i)}_t}$

Returns are sampled every $\delta$ periods.

### Portfolio construction

The portfolio is equal-weighted across `d` selected funds:

$w_i = \frac{1}{d}$

### VaR and CVaR

The dashboard computes historical VaR and CVaR from the empirical distribution of portfolio returns.

### Sector stress testing

Sector shocks are translated into fund-level shocks using the fund sector-exposure matrix:

- sector shock -> fund shock -> portfolio shock

### Bootstrap simulation

The app also resamples portfolio returns with replacement to estimate bootstrap VaR and CVaR.

## Notes

- This project uses historical simulation plus bootstrap resampling.
- Sector stress testing depends on the quality and date coverage of the sector exposure file.
- If the selected date range is too short for the chosen $\tau$, the app will raise an error.
- If too few funds meet the data coverage requirement, reduce `d` or adjust the date range.

