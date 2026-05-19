"""
Generate synthetic HVAC repair dataset calibrated to real-world properties.

The synthetic dataset preserves these key statistical properties of the
underlying real dataset (which is proprietary):

- 75 monthly observations spanning Jan 2020 - Mar 2026
- HVAC repair costs are zero-inflated (~40% of months recorded $0)
- Right-skewed magnitude when nonzero (lognormal)
- 65 pre-treatment + 10 post-treatment months
- Treatment effect: roughly -$1,800/month in HVAC repair costs
- Non-HVAC repair categories are independent of HVAC (r ≈ 0)
- Mild seasonal patterns in electricity (summer peak) and gas (winter peak)

Output: data/hvac_dataset.csv

Deterministic via seed=42. Re-running produces identical output.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_MONTHS = 75
TREAT_IDX = 65  # June 2025 = first post-treatment month

# Target distributional properties
HVAC_PRE_MEAN, HVAC_PRE_SD, HVAC_PRE_PCT_ZERO = 3234, 6285, 0.462
HVAC_POST_MEAN, HVAC_POST_SD, HVAC_POST_PCT_ZERO = 1500, 2000, 0.40
NONHVAC_PRE_MEAN, NONHVAC_PRE_SD = 13310, 19826
NONHVAC_POST_MEAN = 16376
ELEC_PRE_MEAN, ELEC_PRE_SD = 36250, 12000
ELEC_POST_MEAN = 46876
GAS_PRE_MEAN, GAS_PRE_SD = 7078, 7000


def gen_zero_inflated(n, p_zero, target_mean, target_sd, sub_rng):
    """Generate zero-inflated lognormal values with target overall mean/SD."""
    p_nz = 1 - p_zero
    mean_nz = target_mean / p_nz
    var_nz = (target_sd**2 + target_mean**2) / p_nz - mean_nz**2
    if var_nz <= 0:
        var_nz = mean_nz**2
    sd_nz = np.sqrt(var_nz)
    sigma2 = np.log(1 + (sd_nz**2 / mean_nz**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_nz) - sigma2 / 2
    is_nz = sub_rng.random(n) < p_nz
    mags = sub_rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.where(is_nz, mags, 0)


def gen_lognormal(n, target_mean, target_sd, sub_rng):
    """Generate strictly-positive lognormal values."""
    sigma2 = np.log(1 + (target_sd**2 / target_mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(target_mean) - sigma2 / 2
    return sub_rng.lognormal(mean=mu, sigma=sigma, size=n)


def generate():
    dates = pd.date_range('2020-01-01', periods=N_MONTHS, freq='MS')
    months = np.array([d.month for d in dates])
    years = np.array([d.year for d in dates])
    cooling = ((months >= 5) & (months <= 9)).astype(int)
    covid = ((years == 2020) & (months >= 3)).astype(int)
    post = np.concatenate([np.zeros(TREAT_IDX, dtype=int),
                           np.ones(N_MONTHS - TREAT_IDX, dtype=int)])

    # HVAC repairs (zero-inflated with treatment effect)
    hvac_pre = gen_zero_inflated(TREAT_IDX, HVAC_PRE_PCT_ZERO,
                                  HVAC_PRE_MEAN, HVAC_PRE_SD,
                                  np.random.default_rng(SEED + 1))
    hvac_post = gen_zero_inflated(N_MONTHS - TREAT_IDX, HVAC_POST_PCT_ZERO,
                                   HVAC_POST_MEAN, HVAC_POST_SD,
                                   np.random.default_rng(SEED + 2))
    hvac = np.concatenate([hvac_pre, hvac_post])

    # Non-HVAC repairs (independent control series)
    nh_pre = gen_lognormal(TREAT_IDX, NONHVAC_PRE_MEAN, NONHVAC_PRE_SD,
                           np.random.default_rng(SEED + 3))
    nh_post = gen_lognormal(N_MONTHS - TREAT_IDX, NONHVAC_POST_MEAN,
                            NONHVAC_PRE_SD, np.random.default_rng(SEED + 4))
    nonhvac = np.concatenate([nh_pre, nh_post])

    # Electricity (no treatment effect, mild seasonal multiplier)
    elec = gen_lognormal(N_MONTHS, ELEC_PRE_MEAN, ELEC_PRE_SD,
                         np.random.default_rng(SEED + 5))
    elec[post == 1] *= (ELEC_POST_MEAN / ELEC_PRE_MEAN)
    elec *= (1 + 0.20 * cooling)

    # Gas (winter peak)
    gas = gen_lognormal(N_MONTHS, GAS_PRE_MEAN, GAS_PRE_SD,
                        np.random.default_rng(SEED + 6))
    gas *= (1 + 0.5 * (1 - cooling))

    df = pd.DataFrame({
        'month_index': range(1, N_MONTHS + 1),
        'year': years,
        'month': months,
        'hvac_repairs': np.round(hvac).astype(int),
        'electricity': np.round(elec).astype(int),
        'gas': np.round(gas).astype(int),
        'nonhvac_repairs': np.round(nonhvac).astype(int),
        'cooling_season': cooling,
        'covid_period': covid,
        'post_treatment': post,
    })
    return df


if __name__ == '__main__':
    df = generate()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hvac_dataset.csv')
    out_path = os.path.normpath(out_path)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    pre = df[df.post_treatment == 0]
    post = df[df.post_treatment == 1]
    print(f"Pre HVAC : n={len(pre)}, mean=${pre['hvac_repairs'].mean():,.0f}, sd=${pre['hvac_repairs'].std():,.0f}")
    print(f"Post HVAC: n={len(post)}, mean=${post['hvac_repairs'].mean():,.0f}, sd=${post['hvac_repairs'].std():,.0f}")
    print(f"Treatment effect: ${post['hvac_repairs'].mean() - pre['hvac_repairs'].mean():,.0f}/mo")
