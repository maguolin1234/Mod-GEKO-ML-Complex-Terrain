# Mod-GEKO-ML: data, trained model, and OOD analysis materials

This repository contains **only the data, trained GEKO-ML model, and analysis code directly used by the final manuscript**:

> *Improving Wind Flow Simulation over Complex Terrain for Wind Resource Assessment through a Hybrid Data-Driven and Physics-Informed Generalized k-ω Framework*
>
> Physics of Fluids manuscript: **POF26-AR-11012R**

Intermediate revision scripts and unrelated post-processing utilities are intentionally excluded.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── SHA256SUMS.txt
├── model/
│   └── h42_GEKO_NEW_mod3__design_optimal_03_3_02.scm
├── scripts/
│   └── plot_OOD_section6_4_ONLY_Fig17_v34_case_colors_more_distinct.py
└── data/
    ├── NASA_Bump_Family/
    │   ├── LES_bump_family.tar.gz
    │   └── README.md
    ├── Askervein/
    │   ├── askervein_elevation-roughness.map
    │   ├── askervein_inlet1.txt
    │   ├── askervein_sensor1.txt
    │   ├── askervein_validation1.txt
    │   └── README.md
    ├── Bolund/
    │   ├── Bolund_Orography.zip
    │   ├── Bolund_Measurements.zip
    │   └── README.md
    └── ood/
        ├── bolund239/
        │   ├── std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineA
        │   ├── std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineB
        │   └── README.md
        └── h42_reference/
            └── README.md
```

## Trained GEKO-ML model

`model/h42_GEKO_NEW_mod3__design_optimal_03_3_02.scm`

This ANSYS Fluent Scheme file contains the trained reduced-feature GEKO-ML mapping used in the manuscript. The final reduced model uses the retained input features **φ1, φ2, φ6, φ7, and φ8**, a Softsign neural network with hidden-layer topology **[24, 16, 8]**, and outputs **CSEP, CNW, and CMIX**.

## External benchmark data

The repository includes the benchmark data actually used in the manuscript:

- **2-D family of bumps LES data** — NASA/Turbulence Modeling Resource.
- **Askervein Hill input and validation data** — Zenodo record DOI: `10.5281/zenodo.4095052`.
- **Bolund Island orography and measurements** — DTU Bolund blind-comparison database.

The files are kept in their supplied form. See the README inside each benchmark directory for provenance. Third-party data remain subject to the terms of their original providers.

## OOD / transferability analysis

The manuscript evaluates the retained five-dimensional GEKO-ML feature vector relative to the H42 training distribution using a percentile-normalized Mahalanobis distance. In the final manuscript this result appears as **Fig. 15** for the Bolund Island 239° inflow case, Lines A and B.

The retained analysis script is:

`scripts/plot_OOD_section6_4_ONLY_Fig17_v34_case_colors_more_distinct.py`

Its filename preserves the earlier revision-stage section/figure numbering. The paper's final numbering is Fig. 15.

The Bolund 239° Fluent feature exports used by that analysis are included under `data/ood/bolund239/`.

### One remaining author-generated input

To reproduce Fig. 15 completely from scratch, the analysis still requires the **H42 cell-wise feature-space reference export**:

`h42_GEKO_NEW_mod3_design_optimal_03_02_02_data2`

This file is different from both the NASA LES benchmark archive and the trained `.scm` model. It contains the H42 GEKO feature distribution used to construct the Mahalanobis reference distribution. Place it in `data/ood/h42_reference/` when available.

## Python dependencies

```bash
pip install -r requirements.txt
```

## Integrity

`SHA256SUMS.txt` records SHA-256 hashes for the repository data/model/code files so that uploaded copies can be checked against the packaged version.
