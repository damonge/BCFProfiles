# Reggie Dunsdon — summer 2026

Analytic baryon correction (BFC) predictions for ⟨bPe⟩ and its redshift
evolution, compared against FLAMINGO and the DESI BGS/LRG measurements.

## Contents

- `bpe_zevolution.ipynb` — Does the BFC model reproduce FLAMINGO's ⟨bPe⟩(z)
  when fed the FLAMINGO-fitted parameters? Uses Michael's code only, since
  it exposes all three non-thermal parameters. Also compares Y–M200c
  against the FLAMINGO SOAP catalogue.
- `analytical_solution.ipynb` — The likelihood is linear in the template
  parameters (prediction = matrix × params), so ⟨bPe⟩ has a closed-form
  generalised-least-squares solution and doesn't need sampling. Compared
  against the MCMC results as a check.
- `flamingo.py` — `Flamingo` class for loading and plotting FLAMINGO spectra
  and profiles.

## Dependencies

From the repo root: `bfc_pressure_profile_quicker`, `bfc_functions`
(Michael's). From `david_sandbox/`: `utils` (concentration relation).

External: `pyccl`, `numpy`, `scipy`, `matplotlib`, `swiftsimio`, `cobaya`,
`getdist`, `pyyaml`.

## Data

Set these environment variables before running:

| Variable | Points to |
|---|---|
| `FLAM_DATA` | FLAMINGO spectra/profiles (e.g. `.../FLAMINGO/L1_m9`) |
| `FLAM_ANALYSIS` | FLAMINGO SOAP catalogues (`L1000N1800/HYDRO_FIDUCIAL/...`) |
| `BPE_DATA` | Derived products, e.g. `soap_fid_77.npz` |
| `BPE_CHAINS` | MCMC chains from the DESI×tSZ analysis |

All live on glamdring and are not in this repository. The FLAMINGO products
are shared group data. The chains come from the measurement pipeline, which is in a separate private repo; ask me for access.   

## Notes

- `bpe_zevolution.ipynb` applies `SCALE = 1.12` to the BFC prediction to
  match the FLAMINGO normalisation. The origin of this offset is unresolved.
- Loading the SOAP pickle requires a monkeypatch to `swiftsimio`'s
  `cosmo_array.__setstate__`: the file was written with a version storing
  three metadata fields where the current version expects four.
- The measured values plotted against the model (BGS at z = 0.16, 0.32;
  LRG at z = 0.47–0.93) are from the kmax = 0.3, order-2 analysis.

