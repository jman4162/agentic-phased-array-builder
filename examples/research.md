# Phased Array Optimization Protocol

## Objective
Maximize EIRP while keeping hardware cost under $10,000 and maintaining
adequate link margin for a 200 m urban 5G deployment.

## Metric
Primary: eirp_dbw (maximize)
Constraint: cost_usd < 10000, snr_db > 15

## Design Variables
- Array size: Nx (4-16), Ny (4-16)
- Element spacing: dx_m (0.003-0.007), dy_m (0.003-0.007)
- Taper: uniform, taylor, chebyshev, hamming
- TX power per element: 0.01-0.5 W
- Frequency: 28-30 GHz

## Strategy
- Start with the baseline config from apab.yaml
- First sweep taper functions (uniform, taylor, chebyshev) at baseline array size
- Then explore array size variations with the best taper
- Finally tune TX power to maximize EIRP within the cost constraint
- If improvement stalls for 3 consecutive experiments, try a different variable
- Prefer designs with lower cost when EIRP is similar (within 1 dB)
