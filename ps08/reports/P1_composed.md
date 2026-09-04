# Report — approach `P1_composed`

_Shapiro-Wilk self-check on benchmark (n=45): W=0.9852, p=0.8262 — our SW matches the reference (~0.985)._
**Overall mean Shapiro-W across pairs = 0.837** (benchmark target ~0.985).


## GEO (GEO) — test n=69

| channel | W | p | H0 rejected (non-normal)? | mean | std |
|---|---|---|---|---|---|
| x_error (m) | 0.870 | 0.000 | yes | -0.116 | 13.830 |
| y_error (m) | 0.821 | 0.000 | yes | -1.502 | 19.618 |
| z_error (m) | 0.890 | 0.000 | yes | +1.574 | 10.419 |
| satclockerror (m) | 0.578 | 0.000 | yes | -1.029 | 15.863 |

**GEO mean W = 0.790**

## MEO1 (MEO) — test n=6

| channel | W | p | H0 rejected (non-normal)? | mean | std |
|---|---|---|---|---|---|
| x_error (m) | 0.856 | 0.175 | no | +0.048 | 0.218 |
| y_error (m) | 0.981 | 0.958 | no | +0.021 | 0.258 |
| z_error (m) | 0.844 | 0.141 | no | -1.044 | 0.303 |
| satclockerror (m) | 0.948 | 0.723 | no | +0.191 | 0.095 |

**MEO1 mean W = 0.907**

## MEO2 (MEO) — test n=18

| channel | W | p | H0 rejected (non-normal)? | mean | std |
|---|---|---|---|---|---|
| x_error (m) | 0.856 | 0.011 | yes | +0.127 | 0.225 |
| y_error (m) | 0.862 | 0.013 | yes | -0.049 | 0.054 |
| z_error (m) | 0.832 | 0.004 | yes | -0.184 | 0.223 |
| satclockerror (m) | 0.710 | 0.000 | yes | +0.005 | 0.051 |

**MEO2 mean W = 0.815**
