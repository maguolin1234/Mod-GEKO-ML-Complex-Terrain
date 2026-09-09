#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 6.4 ONLY: feature-space OOD / transferability analysis
==============================================================

This script is intentionally limited to Sec. 6.4.
It does NOT read or reproduce the Sec. 6.3 RMSE/component-attribution table.

Inputs
------
Nine Fluent ASCII exports in one folder:

1) h42_GEKO_NEW_mod3_design_optimal_03_02_02_data2
2) h38_GEKO_NEW_mod3_design_optimal_03_02_02_data2
3) h31_GEKO_NEW_mod3_design_optimal_03_02_02_data2
4) std_ask_improve_03_default_geko_h42_design_optional_data2_A
5) std_ask_improve_03_default_geko_h42_design_optional_data2_AA
6) std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_270_lineA
7) std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_270_lineB
8) std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineA
9) std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineB

Main Fig. 17
------------
(a) Case-level normalized feature-distribution shift relative to H42.
(b) Feature-wise fraction outside the H42 robust support.
(c) Feature-wise normalized Wasserstein distribution shift.
(d) Joint phi7-phi8 feature-space comparison, showing the structured
    atmospheric-scale shift relative to the H42 robust support.

Primary OOD definitions
-----------------------
- H42 is used as the calibration-case feature-space reference.
- Retained NN inputs:
    phi1 = non-equilibrium-parameter
    phi2 = second-invariant
    phi6 = length-ratio
    phi7 = turbulent-reynolds-number-scaled
    phi8 = turbulent-viscosity-ratio-scaled
- H42 robust support:
    [Q0.5%, Q99.5%] feature-by-feature.
- Normalized per-feature Wasserstein shift:
    W*_j = W1(P_H42,j, P_target,j) / IQR_H42,j
- Overall shift:
    W*_OOD = mean_j(W*_j)
- For field cases, Line A / Line AA or Line A / Line B are equally weighted
  at case level. Raw cell counts are NOT used as inter-transect weights.

Field sampling windows
----------------------
Askervein:
    |s - s_crest| <= 1000 m
    z_AGL <= 200 m
    The crest location is estimated automatically from the reconstructed
    lower terrain envelope of each plane.

Bolund:
    -100 <= s <= 130 m relative to CP
    z_AGL <= 20 m

The script also writes sensitivity statistics for alternative vertical
and horizontal windows.

Required packages
-----------------
numpy, pandas, scipy, matplotlib

Run
---
python plot_OOD_section6_4_ONLY_Fig17.py

or

python plot_OOD_section6_4_ONLY_Fig17.py ^
    --root "H:\\2022.3\\paper\\GEKO湍流模型\\figure\\07"

Outputs
-------
OOD_section6_4/
    01_summary/
    02_figures/
    03_supplementary/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

DEFAULT_ROOT = Path(r"H:\2022.3\paper\GEKO湍流模型\figure\07")
DEFAULT_OUTPUT_NAME = "OOD_section6_4"

# Main terrain-focused windows.
ASK_HALF_STREAMWISE_M = 1000.0
ASK_ZAGL_MAX_M = 200.0

BOL_S_MIN_M = -100.0
BOL_S_MAX_M = 130.0
BOL_ZAGL_MAX_M = 20.0

# Terrain reconstruction.
ASK_TERRAIN_BIN_M = 5.0
BOL_TERRAIN_BIN_M = 0.5
TERRAIN_SMOOTH_WINDOW = 5

# H42 robust support.
SUPPORT_Q_LOW = 0.005
SUPPORT_Q_HIGH = 0.995

# Mahalanobis reference / diagnostics.
MAHALANOBIS_DIAG_REG = 1.0e-8
MAHALANOBIS_REF_Q = 0.99  # retained only for the secondary Q99-scaled diagnostic

# Sensitivity tests.
ASK_Z_SENSITIVITY = [100.0, 150.0, 200.0]
ASK_S_SENSITIVITY = [750.0, 1000.0, 1500.0]
BOL_Z_SENSITIVITY = [10.0, 15.0, 20.0, 30.0, 40.0]


# Region definitions used for the additional OOD summary figure.
# These ranges are intentionally explicit and easy to modify.
ASK_REGION_BOUNDS = {
    "Windward": (-1000.0, -200.0),
    "Crest": (-200.0, 200.0),
    "Leeward slope": (200.0, 1000.0),
}

BOL_REGION_BOUNDS = {
    "Pre-cliff": (-100.0, 0.0),
    "Immediate post-cliff": (0.0, 30.0),
    "Central plateau": (30.0, 80.0),
    "Lee-side": (80.0, 130.0),
}

# Manual GEKO-ML incremental U-performance for Sec. 6.4 panel (g).
# These values come from the regional RMSE table already prepared for Sec. 6.3.
# Definition:
#   G_ML,U = (RMSE_Std-GEKO^2 - RMSE_GEKO-ML^2) / RMSE_Std-GEKO^2 * 100%
REGIONAL_ML_GAIN_U = {
    ("Askervein", "Leeward slope"): +32.4,
    ("Bolund 270°", "Immediate post-cliff"): -18.5,
    ("Bolund 270°", "Central plateau"): +35.9,
    ("Bolund 270°", "Lee-side"): -12.6,
    ("Bolund 239°", "Immediate post-cliff"): -5.9,
    ("Bolund 239°", "Central plateau"): +8.6,
    ("Bolund 239°", "Lee-side"): -28.1,
}

# Plotting.
SAVE_DPI = 1200
FIGSIZE = (14.0, 10.5)
FONT_FAMILY = "Times New Roman"
RANDOM_SEED = 42


# =============================================================================
# FINAL FIG.17 (2 x 3) SETTINGS — Section 6.4 main-text figure
# =============================================================================
# Layout requested by the PoF revision plan:
#   (a-c) spatial multivariate OOD maps for the three atmospheric field cases
#   (d)   regional OOD median + IQR
#   (e)   regional OOD vs ML-only U RMSE change
#   (f)   regional OOD vs ML-only TKE RMSE change
# H38/H31 are still computed elsewhere, but do not occupy the main spatial row.
FINAL_FIG_DPI = 1800
FINAL_FIGSIZE = (14.6, 10.6)
FINAL_CMAP = "viridis"
FINAL_MAHAL_VMIN = 0.0
FINAL_MAHAL_VMAX = 1.0
FINAL_HIGH_MAHAL_THRESHOLD = 1.0

# Spatial-map display range: shared across all six atmospheric transects.
# The range is determined from the pooled atmospheric D_M* distribution so
# that the spatial contrast is visible. Values outside the range are clipped
# only for visualization; the underlying statistics are unchanged.
FINAL_MAHAL_DISPLAY_QUANTILES = (0.02, 0.98)
FINAL_MAHAL_DISPLAY_ROUND = 0.05

# Spatial-panel vertical display limits requested by the user.
# Askervein: show 0-200 m; Bolund: show 0-20 m.
FINAL_SPATIAL_YMAX_BY_KEY = {
    "ASK_A": 200.0,
    "ASK_AA": 200.0,
    "B270_A": 20.0,
    "B270_B": 20.0,
    "B239_A": 20.0,
    "B239_B": 20.0,
}


# =============================================================================
# VALIDATION-LOCATION SUPPORT FOR FIG.17(d-f)
# =============================================================================
# Panels (d-f) are evaluated at the SAME measurement/mast locations used by
# the regional U/TKE error statistics, rather than over all CFD cells in each
# terrain region.
#
# IMPORTANT (v14): Askervein locations are fully hard-coded below. The script
# DOES NOT read OBS.xlsx at runtime.
#
# Geometry references are already defined in REFERENCE_POINTS_XY:
#   HT = (75669.70,   23824.66)     -> ASK_A  s_raw_m = 0 at HT
#   CP = (75586.7266, 23437.5957)   -> ASK_AA s_raw_m = 0 at CP
#
# All Askervein validation samples are evaluated at 10 m ABOVE LOCAL TERRAIN
# (z_AGL = 10 m), not at absolute z = 10 m.
ASK_VALIDATION_Z_AGL_M = 10.0

# Askervein Line A: 9 validation locations, signed distance from HT [m].
# These locations are used for BOTH U and TKE.
ASK_LINE_A_VALIDATION = [
    # point, physical mast, s_from_HT_m
    ("A1", "ASW85", -841.0),
    ("A2", "ASW50", -492.0),
    ("A3", "ASW35", -327.0),
    ("A4", "ASW20", -186.0),
    ("A5", "ASW10",  -96.0),
    ("A6", "HT",       0.0),
    ("A7", "ANE10",  100.0),
    ("A8", "ANE20",  198.0),
    ("A9", "ANE40",  393.0),
]

# Askervein Line AA: 15 U-validation locations, signed distance from CP [m],
# in the same AA10-AA24 order used by the manuscript regional U statistics.
ASK_LINE_AA_U_VALIDATION = [
    ("AA10", -684.07769),
    ("AA11", -585.96880),
    ("AA12", -489.59328),
    ("AA13", -388.36433),
    ("AA14", -284.36197),
    ("AA15", -184.86638),
    ("AA16", -136.67863),
    ("AA17",  -85.37080),
    ("AA18",    0.95116),
    ("AA19",   -0.43554),
    ("AA20",  104.95351),
    ("AA21",  203.06240),
    ("AA22",  292.15775),
    ("AA23",  388.87994),
    ("AA24",  583.36434),
]

# Askervein Line AA: sparse TKE observations.  The manuscript regional TKE
# table counts these three AA measurements as AA10-AA12; their actual physical
# mast positions are AASW10/AASW30/AASW50 below.  All are windward.
ASK_LINE_AA_TKE_VALIDATION = [
    # point_id_in_table, physical_mast, s_from_CP_m
    ("AA10", "AASW10",  -93.0),
    ("AA11", "AASW30", -287.0),
    ("AA12", "AASW50", -489.0),
]

# Exact Bolund projected mast distances from CP along the two experimental
# transects (m). M3 belongs to both transects; its OOD value is averaged from
# the two plane extractions at the same physical measurement height.
BOLUND_MAST_S_BY_LINE = {
    "A": {"M1": -60.9, "M2": -40.7, "M3": 2.8, "M4": 59.9},
    "B": {"M7": -66.9, "M6": -46.1, "M3": 3.2, "M8": 92.0},
}

# Measurement-height support used by the regional tables in this manuscript.
# These reproduce the regional point counts:
#   Bolund 270°: 5 / 7 / 3 / 5 = 20
#   Bolund 239°: 5 / 6 / 3 / 5 = 19
BOLUND_HEIGHTS_270 = {
    # Exact sonic-anemometer AGL heights used by the 20-point comparison.
    # M2 contributes five heights; M6 contributes two heights.
    "M1": [2.1, 5.1, 9.0],
    "M7": [2.0, 5.0],
    "M2": [1.1, 2.1, 3.6, 5.1, 9.1],
    "M6": [1.9, 4.9],
    "M3": [2.0, 5.0, 9.0],
    "M4": [1.4, 4.4, 8.4],
    "M8": [1.8, 4.7],
}
BOLUND_HEIGHTS_239 = {
    # Exact sonic-anemometer AGL heights for the 19 retained observations.
    # The M2 3.6-m sonic record is not part of the 239° 19-point table,
    # hence M2 contributes 1.1, 2.1, 5.1 and 9.1 m only.
    "M1": [2.1, 5.1, 9.0],
    "M7": [2.0, 5.0],
    "M2": [1.1, 2.1, 5.1, 9.1],
    "M6": [1.9, 4.9],
    "M3": [2.0, 5.0, 9.0],
    "M4": [1.4, 4.4, 8.4],
    "M8": [1.8, 4.7],
}

# Regional grouping exactly matching the regional error statistics.
ASK_REGION_FROM_POINT = {
    **{f"A{i}": "Windward" for i in range(1, 6)},
    "A6": "Crest",
    **{f"A{i}": "Lee" for i in range(7, 10)},
    **{f"AA{i}": "Windward" for i in range(10, 17)},
    **{f"AA{i}": "Crest" for i in range(17, 21)},
    **{f"AA{i}": "Lee" for i in range(21, 25)},
}
BOLUND_REGION_FROM_MAST = {
    "M1": "Pre-cliff", "M7": "Pre-cliff",
    "M2": "Post-cliff", "M6": "Post-cliff",
    "M3": "Plateau", "M4": "Lee", "M8": "Lee",
}


# Expected regional sample counts in the manuscript tables.
EXPECTED_BOLUND_REGION_COUNTS = {
    "Bolund 270°": {"Pre-cliff": 5, "Post-cliff": 7, "Plateau": 3, "Lee": 5},
    "Bolund 239°": {"Pre-cliff": 5, "Post-cliff": 6, "Plateau": 3, "Lee": 5},
}

# Validation-aligned vertical planes for the spatial row.
# Each atmospheric case is shown with BOTH available validation transects.
# The outer panel labels remain (a)-(c), while each panel contains two
# vertically stacked transect maps.
FINAL_SPATIAL_GROUPS = [
    ("Askervein Hill", [("ASK_A", "Line A"), ("ASK_AA", "Line AA")]),
    ("Bolund 270°", [("B270_A", "Line A"), ("B270_B", "Line B")]),
    ("Bolund 239°", [("B239_A", "Line A"), ("B239_B", "Line B")]),
]

FINAL_SPATIAL_BINS = {
    "ASK_A":  (210, 120),
    "ASK_AA": (210, 120),
    "B270_A": (190, 105),
    "B270_B": (190, 105),
    "B239_A": (190, 105),
    "B239_B": (190, 105),
}

# Regional streamwise bounds used for the final Sec. 6.4 regional analysis.
# Askervein limits are relative to each automatically detected crest.
# Bolund limits use the mast-midpoint boundaries established for Line A/B.
FINAL_REGION_BOUNDS_BY_TRANSECT = {
    "ASK_A": [
        ("Windward", -1000.0, -200.0),
        ("Crest",      -200.0,  200.0),
        ("Lee",         200.0, 1000.0),
    ],
    "ASK_AA": [
        ("Windward", -1000.0, -200.0),
        ("Crest",      -200.0,  200.0),
        ("Lee",         200.0, 1000.0),
    ],
    "B270_A": [
        ("Pre-cliff",  -100.0, -50.8),
        ("Post-cliff",  -50.8, -18.95),
        ("Plateau",     -18.95, 31.35),
        ("Lee",          31.35, 130.0),
    ],
    "B239_A": [
        ("Pre-cliff",  -100.0, -50.8),
        ("Post-cliff",  -50.8, -18.95),
        ("Plateau",     -18.95, 31.35),
        ("Lee",          31.35, 130.0),
    ],
    "B270_B": [
        ("Pre-cliff",  -100.0, -56.5),
        ("Post-cliff",  -56.5, -21.45),
        ("Plateau",     -21.45, 47.6),
        ("Lee",          47.6, 130.0),
    ],
    "B239_B": [
        ("Pre-cliff",  -100.0, -56.5),
        ("Post-cliff",  -56.5, -21.45),
        ("Plateau",     -21.45, 47.6),
        ("Lee",          47.6, 130.0),
    ],
}

# Regional Std-GEKO -> GEKO-ML RMSE pairs used only for Fig.17(e,f).
# The code converts each pair to the controlled ML-only absolute RMSE change:
# DeltaRMSE_ML = RMSE_Std - RMSE_ML.
# Positive values indicate RMSE reduction (improvement); negative values indicate degradation.
FINAL_ML_RMSE = {
    ("Askervein", "Windward", "U"):   (0.459, 0.490),
    ("Askervein", "Crest", "U"):      (0.888, 0.631),
    ("Askervein", "Lee", "U"):        (1.034, 0.850),
    ("Askervein", "Windward", "TKE"): (0.163, 0.163),
    ("Askervein", "Crest", "TKE"):    (0.116, 0.123),
    ("Askervein", "Lee", "TKE"):      (1.955, 1.772),

    ("Bolund 270°", "Pre-cliff", "U"):    (1.245, 1.523),
    ("Bolund 270°", "Post-cliff", "U"):   (2.661, 2.897),
    ("Bolund 270°", "Plateau", "U"):      (0.512, 0.410),
    ("Bolund 270°", "Lee", "U"):          (0.768, 0.815),
    ("Bolund 270°", "Pre-cliff", "TKE"):  (4.014, 4.553),
    ("Bolund 270°", "Post-cliff", "TKE"): (6.630, 6.784),
    ("Bolund 270°", "Plateau", "TKE"):    (0.636, 0.744),
    ("Bolund 270°", "Lee", "TKE"):        (1.412, 1.236),

    ("Bolund 239°", "Pre-cliff", "U"):    (0.194, 0.194),
    ("Bolund 239°", "Post-cliff", "U"):   (1.607, 1.654),
    ("Bolund 239°", "Plateau", "U"):      (0.663, 0.634),
    ("Bolund 239°", "Lee", "U"):          (0.744, 0.842),
    ("Bolund 239°", "Pre-cliff", "TKE"):  (0.519, 0.505),
    ("Bolund 239°", "Post-cliff", "TKE"): (3.207, 2.713),
    ("Bolund 239°", "Plateau", "TKE"):    (1.194, 1.192),
    ("Bolund 239°", "Lee", "TKE"):        (1.614, 1.451),
}

# Max points used only for panel (d) visualization.
# Statistics always use all selected points.
PANEL_D_H42_N = 7000
PANEL_D_BUMP_N = 2200
PANEL_D_FIELD_PER_TRANSECT_N = 2200


# =============================================================================
# 2. INPUT FILE NAMES
# =============================================================================

FILE_NAMES = {
    "H42": "h42_GEKO_NEW_mod3_design_optimal_03_02_02_data2",
    "H38": "h38_GEKO_NEW_mod3_design_optimal_03_02_02_data2",
    "H31": "h31_GEKO_NEW_mod3_design_optimal_03_02_02_data2",

    "ASK_A": "std_ask_improve_03_default_geko_h42_design_optional_data2_A",
    "ASK_AA": "std_ask_improve_03_default_geko_h42_design_optional_data2_AA",

    "B270_A": "std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_270_lineA",
    "B270_B": "std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_270_lineB",

    "B239_A": "std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineA",
    "B239_B": "std_bol_07-mesh_HEIGH_default_geko_h42_design_opt_data2_239_lineB",
}

# Reference points used only to define the initial along-transect coordinate s.
# The actual Askervein terrain window is re-centered on the automatically
# estimated crest before OOD statistics are evaluated.
REFERENCE_POINTS_XY = {
    "ASK_A": (75669.70, 23824.66),       # HT
    "ASK_AA": (75586.7266, 23437.5957), # CP/reference used in AA processing
    "B270_A": (0.0, 0.0),
    "B270_B": (0.0, 0.0),
    "B239_A": (0.0, 0.0),
    "B239_B": (0.0, 0.0),
}


# =============================================================================
# 3. FEATURE DEFINITIONS
# =============================================================================

FEATURES = [
    ("phi1", "non-equilibrium-parameter", r"$\phi_1$"),
    ("phi2", "second-invariant", r"$\phi_2$"),
    ("phi6", "length-ratio", r"$\phi_6$"),
    ("phi7", "turbulent-reynolds-number-scaled", r"$\phi_7$"),
    ("phi8", "turbulent-viscosity-ratio-scaled", r"$\phi_8$"),
]

FEATURE_KEYS = [x[0] for x in FEATURES]
FEATURE_COLS = [x[1] for x in FEATURES]
FEATURE_LABELS = [x[2] for x in FEATURES]


# =============================================================================
# 4. DATA STRUCTURES
# =============================================================================

@dataclass
class H42Reference:
    values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray
    q25: np.ndarray
    q50: np.ndarray
    q75: np.ndarray
    iqr: np.ndarray
    cov_z: np.ndarray
    cov_z_inv: np.ndarray
    mahal_p99: float


@dataclass
class PlaneData:
    key: str
    df: pd.DataFrame
    direction_xy: np.ndarray
    s_raw: np.ndarray
    terrain_z: np.ndarray
    z_agl: np.ndarray
    crest_s_raw: float


# =============================================================================
# 5. READ / CLEAN
# =============================================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def read_ascii(path: Path, require_z: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Fluent ASCII exported in the current workflow is comma-separated.
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df = clean_columns(df)

    required = {"cellnumber", "x-coordinate", "y-coordinate", *FEATURE_COLS}
    if require_z:
        required.add("z-coordinate")

    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"{path.name}: missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    numeric_cols = ["cellnumber", "x-coordinate", "y-coordinate", *FEATURE_COLS]
    if "z-coordinate" in df.columns:
        numeric_cols.append("z-coordinate")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    finite_cols = ["x-coordinate", "y-coordinate", *FEATURE_COLS]
    if require_z:
        finite_cols.append("z-coordinate")

    mask = np.ones(len(df), dtype=bool)
    for c in finite_cols:
        mask &= np.isfinite(df[c].to_numpy(dtype=float))

    n0 = len(df)
    df = df.loc[mask].copy().reset_index(drop=True)
    ndrop = n0 - len(df)
    if ndrop:
        print(f"[WARN] {path.name}: dropped {ndrop:,} non-finite rows")

    ndups = int(df["cellnumber"].duplicated().sum())
    if ndups:
        print(f"[WARN] {path.name}: duplicated cellnumber rows = {ndups:,}")

    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


# =============================================================================
# 6. FIELD-PLANE COORDINATES / TERRAIN / AGL
# =============================================================================

def infer_transect_direction(df: pd.DataFrame) -> np.ndarray:
    """
    Infer horizontal line direction using PCA.
    Orient the direction so s generally increases with global x.
    """
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(dtype=float)
    center = np.mean(xy, axis=0)
    X = xy - center

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    direction = direction / np.linalg.norm(direction)

    proj = X @ direction
    if np.std(proj) > 0 and np.std(xy[:, 0]) > 0:
        corr = np.corrcoef(proj, xy[:, 0])[0, 1]
        if np.isfinite(corr) and corr < 0:
            direction = -direction
    elif direction[0] < 0:
        direction = -direction

    return direction


def compute_s_raw(
    df: pd.DataFrame,
    reference_xy: Tuple[float, float],
    direction_xy: np.ndarray,
) -> np.ndarray:
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(dtype=float)
    return (xy - np.asarray(reference_xy, dtype=float)) @ direction_xy


def reconstruct_terrain_lower_envelope(
    s: np.ndarray,
    z: np.ndarray,
    bin_width: float,
    smooth_window: int,
) -> np.ndarray:
    """
    Estimate terrain z(s) from the minimum cell-center z in streamwise bins.
    This is used only to obtain a consistent lower-layer z_AGL coordinate.
    """
    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)

    smin = float(np.nanmin(s))
    smax = float(np.nanmax(s))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        raise ValueError("Invalid s range for terrain reconstruction.")

    edges = np.arange(smin, smax + bin_width, bin_width)
    if len(edges) < 3:
        edges = np.linspace(smin, smax, 3)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ibin = np.digitize(s, edges) - 1

    terrain = np.full(len(centers), np.nan, dtype=float)
    for i in range(len(centers)):
        zz = z[ibin == i]
        zz = zz[np.isfinite(zz)]
        if len(zz):
            terrain[i] = np.min(zz)

    good = np.isfinite(terrain)
    if good.sum() < 2:
        raise RuntimeError("Too few valid terrain-envelope bins.")

    terrain = np.interp(centers, centers[good], terrain[good])

    if smooth_window > 1:
        terrain = (
            pd.Series(terrain)
            .rolling(int(smooth_window), center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )

    return np.interp(s, centers, terrain, left=terrain[0], right=terrain[-1])


def estimate_askervein_crest(
    s_raw: np.ndarray,
    terrain_z: np.ndarray,
) -> float:
    """
    Estimate crest position from the reconstructed terrain envelope.
    Search near the transect reference rather than over the full several-km plane.
    """
    mask = np.isfinite(s_raw) & np.isfinite(terrain_z) & (np.abs(s_raw) <= 1000.0)
    if mask.sum() < 10:
        mask = np.isfinite(s_raw) & np.isfinite(terrain_z)

    s = s_raw[mask]
    tz = terrain_z[mask]

    # Use all locations within a tiny tolerance of the maximum and take median s,
    # reducing sensitivity to a flat crest represented by several bins.
    zmax = np.nanmax(tz)
    tol = max(0.05, 0.001 * max(1.0, np.nanmax(tz) - np.nanmin(tz)))
    near = tz >= zmax - tol

    if np.any(near):
        return float(np.nanmedian(s[near]))
    return float(s[np.nanargmax(tz)])


def build_plane_data(key: str, df: pd.DataFrame) -> PlaneData:
    if key not in REFERENCE_POINTS_XY:
        raise KeyError(f"Reference point not configured for {key}")

    direction = infer_transect_direction(df)
    s_raw = compute_s_raw(df, REFERENCE_POINTS_XY[key], direction)

    z = df["z-coordinate"].to_numpy(dtype=float)
    bin_width = ASK_TERRAIN_BIN_M if key.startswith("ASK_") else BOL_TERRAIN_BIN_M
    terrain_z = reconstruct_terrain_lower_envelope(
        s_raw, z, bin_width, TERRAIN_SMOOTH_WINDOW
    )
    z_agl = z - terrain_z

    if key.startswith("ASK_"):
        crest = estimate_askervein_crest(s_raw, terrain_z)
    else:
        crest = 0.0

    print(
        f"[PLANE] {key:7s} "
        f"N={len(df):,} "
        f"dir=({direction[0]:.6f},{direction[1]:.6f}) "
        f"s_raw=[{np.nanmin(s_raw):.1f},{np.nanmax(s_raw):.1f}] m "
        f"crest_s={crest:.2f} m "
        f"zAGL=[{np.nanmin(z_agl):.3f},{np.nanmax(z_agl):.1f}] m"
    )

    return PlaneData(
        key=key,
        df=df,
        direction_xy=direction,
        s_raw=s_raw,
        terrain_z=terrain_z,
        z_agl=z_agl,
        crest_s_raw=crest,
    )


def select_plane_window(
    plane: PlaneData,
    s_min: float,
    s_max: float,
    z_agl_max: float,
    center_on_crest: bool,
    z_agl_min: float = -1.0,
) -> pd.DataFrame:
    if center_on_crest:
        s_eval = plane.s_raw - plane.crest_s_raw
    else:
        s_eval = plane.s_raw.copy()

    mask = (
        np.isfinite(s_eval)
        & np.isfinite(plane.z_agl)
        & (s_eval >= s_min)
        & (s_eval <= s_max)
        & (plane.z_agl >= z_agl_min)
        & (plane.z_agl <= z_agl_max)
    )

    out = plane.df.loc[mask].copy().reset_index(drop=True)
    out["s_m"] = s_eval[mask]
    out["s_raw_m"] = plane.s_raw[mask]
    out["terrain_z_m"] = plane.terrain_z[mask]
    out["z_agl_m"] = plane.z_agl[mask]
    return out


def select_main_field_window(plane: PlaneData) -> pd.DataFrame:
    if plane.key.startswith("ASK_"):
        return select_plane_window(
            plane,
            -ASK_HALF_STREAMWISE_M,
            +ASK_HALF_STREAMWISE_M,
            ASK_ZAGL_MAX_M,
            center_on_crest=True,
        )
    return select_plane_window(
        plane,
        BOL_S_MIN_M,
        BOL_S_MAX_M,
        BOL_ZAGL_MAX_M,
        center_on_crest=False,
    )


# =============================================================================
# 7. H42 REFERENCE
# =============================================================================

def mahalanobis_values_from_z(Z: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, cov_inv, Z))


def build_h42_reference(h42_df: pd.DataFrame) -> H42Reference:
    X = feature_matrix(h42_df)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)

    q_low = np.quantile(X, SUPPORT_Q_LOW, axis=0)
    q_high = np.quantile(X, SUPPORT_Q_HIGH, axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    q50 = np.quantile(X, 0.50, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr > 0, iqr, 1.0)

    Z = (X - mean[None, :]) / std[None, :]
    cov_z = np.cov(Z, rowvar=False, ddof=1)
    cov_reg = cov_z + MAHALANOBIS_DIAG_REG * np.eye(len(FEATURE_COLS))
    cov_inv = np.linalg.inv(cov_reg)

    d = mahalanobis_values_from_z(Z, cov_inv)
    mahal_p99 = float(np.quantile(d, MAHALANOBIS_REF_Q))

    print("[H42] standardized covariance condition number:",
          f"{np.linalg.cond(cov_z):.6f}")
    print("[H42] Mahalanobis Q99:", f"{mahal_p99:.9f}")

    return H42Reference(
        values=X,
        mean=mean,
        std=std,
        q_low=q_low,
        q_high=q_high,
        q25=q25,
        q50=q50,
        q75=q75,
        iqr=iqr,
        cov_z=cov_z,
        cov_z_inv=cov_inv,
        mahal_p99=mahal_p99,
    )


def h42_reference_table(ref: H42Reference) -> pd.DataFrame:
    rows = []
    for j, (key, col, _) in enumerate(FEATURES):
        rows.append({
            "feature": key,
            "column": col,
            "mean": ref.mean[j],
            "std": ref.std[j],
            "Q0.5%": ref.q_low[j],
            "Q25%": ref.q25[j],
            "median": ref.q50[j],
            "Q75%": ref.q75[j],
            "Q99.5%": ref.q_high[j],
            "IQR": ref.iqr[j],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 8. OOD METRICS
# =============================================================================

def normalized_wasserstein(
    ref: H42Reference,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    X = feature_matrix(df)
    if len(X) == 0:
        return np.full(len(FEATURE_COLS), np.nan), float("nan")

    w = np.array([
        wasserstein_distance(ref.values[:, j], X[:, j]) / ref.iqr[j]
        for j in range(len(FEATURE_COLS))
    ], dtype=float)

    return w, float(np.mean(w))


def feature_support_exceedance(
    ref: H42Reference,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    X = feature_matrix(df)
    if len(X) == 0:
        return np.full(len(FEATURE_COLS), np.nan), float("nan")

    outside = (X < ref.q_low[None, :]) | (X > ref.q_high[None, :])
    per_feature = np.mean(outside, axis=0) * 100.0
    any_feature = float(np.mean(np.any(outside, axis=1)) * 100.0)
    return per_feature, any_feature


def mahalanobis_summary(
    ref: H42Reference,
    df: pd.DataFrame,
) -> Dict[str, float]:
    X = feature_matrix(df)
    if len(X) == 0:
        return {
            "N": 0,
            "median_Dstar": np.nan,
            "P90_Dstar": np.nan,
            "P95_Dstar": np.nan,
            "P99_Dstar": np.nan,
            "mean_Dstar": np.nan,
            "max_Dstar": np.nan,
            "fraction_Dstar_ge_0p99_pct": np.nan,
            "median_DQ99": np.nan,
            "fraction_DQ99_gt1_pct": np.nan,
        }

    Z = (X - ref.mean[None, :]) / ref.std[None, :]
    raw_d = mahalanobis_values_from_z(Z, ref.cov_z_inv)

    h42_sorted = h42_mahalanobis_reference_sorted(ref)
    dstar = percentile_normalize_mahalanobis(raw_d, h42_sorted)
    dq99 = raw_d / ref.mahal_p99

    return {
        "N": int(len(dstar)),
        "median_Dstar": float(np.quantile(dstar, 0.50)),
        "P90_Dstar": float(np.quantile(dstar, 0.90)),
        "P95_Dstar": float(np.quantile(dstar, 0.95)),
        "P99_Dstar": float(np.quantile(dstar, 0.99)),
        "mean_Dstar": float(np.mean(dstar)),
        "max_Dstar": float(np.max(dstar)),
        "fraction_Dstar_ge_0p99_pct": float(np.mean(dstar >= 0.99) * 100.0),
        "median_DQ99": float(np.quantile(dq99, 0.50)),
        "fraction_DQ99_gt1_pct": float(np.mean(dq99 > 1.0) * 100.0),
    }


def summarize_one_dataset(
    dataset: str,
    df: pd.DataFrame,
    ref: H42Reference,
) -> Dict[str, float]:
    w, wmean = normalized_wasserstein(ref, df)
    exc, any_exc = feature_support_exceedance(ref, df)
    m = mahalanobis_summary(ref, df)

    row: Dict[str, float] = {
        "dataset": dataset,
        "N": int(len(df)),
        "W_OOD_mean": wmean,
        "any_feature_exceedance_pct": any_exc,
    }

    X = feature_matrix(df)
    for j, key in enumerate(FEATURE_KEYS):
        row[f"W_{key}"] = float(w[j])
        row[f"exceed_{key}_pct"] = float(exc[j])
        row[f"median_{key}"] = float(np.median(X[:, j]))

    row.update({k: v for k, v in m.items() if k != "N"})
    return row


# =============================================================================
# 9. EQUAL-TRANSECT CASE SUMMARY
# =============================================================================

def make_case_level_summary(transect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight field transects, not cell-count weighting.
    """
    rows = []

    for case in ["H42", "H38", "H31"]:
        r = transect_df.loc[transect_df["dataset"] == case].iloc[0]
        row = {
            "case": case,
            "n_transects": 1,
            "W_OOD_mean": 0.0 if case == "H42" else float(r["W_OOD_mean"]),
        }
        for key in FEATURE_KEYS:
            row[f"W_{key}"] = 0.0 if case == "H42" else float(r[f"W_{key}"])
        rows.append(row)

    field_groups = {
        "Askervein": ["ASK_A", "ASK_AA"],
        "Bolund 270°": ["B270_A", "B270_B"],
        "Bolund 239°": ["B239_A", "B239_B"],
    }

    for case, members in field_groups.items():
        rr = transect_df[transect_df["dataset"].isin(members)]
        row = {
            "case": case,
            "n_transects": int(len(rr)),
            "W_OOD_mean": float(rr["W_OOD_mean"].mean()),
        }
        for key in FEATURE_KEYS:
            row[f"W_{key}"] = float(rr[f"W_{key}"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 10. SENSITIVITY
# =============================================================================

def sensitivity_statistics(
    planes: Dict[str, PlaneData],
    ref: H42Reference,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    # Askervein: vary both s and z_AGL.
    for key in ["ASK_A", "ASK_AA"]:
        p = planes[key]
        for half_s in ASK_S_SENSITIVITY:
            for zmax in ASK_Z_SENSITIVITY:
                sub = select_plane_window(
                    p, -half_s, +half_s, zmax, center_on_crest=True
                )
                w, wmean = normalized_wasserstein(ref, sub)
                rows.append({
                    "dataset": key,
                    "half_streamwise_window_m": half_s,
                    "s_min_m": -half_s,
                    "s_max_m": half_s,
                    "z_agl_max_m": zmax,
                    "N": len(sub),
                    "W_OOD_mean": wmean,
                    **{f"W_{k}": float(v) for k, v in zip(FEATURE_KEYS, w)},
                })

    # Bolund: fixed streamwise window, varying z_AGL.
    for key in ["B270_A", "B270_B", "B239_A", "B239_B"]:
        p = planes[key]
        for zmax in BOL_Z_SENSITIVITY:
            sub = select_plane_window(
                p, BOL_S_MIN_M, BOL_S_MAX_M, zmax, center_on_crest=False
            )
            w, wmean = normalized_wasserstein(ref, sub)
            rows.append({
                "dataset": key,
                "half_streamwise_window_m": np.nan,
                "s_min_m": BOL_S_MIN_M,
                "s_max_m": BOL_S_MAX_M,
                "z_agl_max_m": zmax,
                "N": len(sub),
                "W_OOD_mean": wmean,
                **{f"W_{k}": float(v) for k, v in zip(FEATURE_KEYS, w)},
            })

    return pd.DataFrame(rows)




def equal_weight_case_mixture(
    selected: Mapping[str, pd.DataFrame],
    members: Sequence[str],
    columns: Sequence[str],
    nq_per_member: int = 25000,
) -> pd.DataFrame:
    """
    Equal-transect mixture for case-level visualization.
    Each member contributes the same quantile-representative sample size,
    avoiding bias from unequal Fluent cell counts.
    """
    qgrid = (np.arange(nq_per_member, dtype=float) + 0.5) / nq_per_member
    out = {}
    for col in columns:
        parts = []
        for member in members:
            vals = selected[member][col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            parts.append(np.quantile(vals, qgrid))
        if not parts:
            out[col] = np.array([], dtype=float)
        else:
            out[col] = np.concatenate(parts)
    return pd.DataFrame(out)


def standardized_case_feature_data(
    selected: Mapping[str, pd.DataFrame],
    ref: H42Reference,
) -> Dict[str, pd.DataFrame]:
    """
    Return case-level equal-weight feature samples standardized by the H42
    mean and standard deviation.
    """
    cases = {
        "H42": ["H42"],
        "H38": ["H38"],
        "H31": ["H31"],
        "Askervein": ["ASK_A", "ASK_AA"],
        "Bolund 270°": ["B270_A", "B270_B"],
        "Bolund 239°": ["B239_A", "B239_B"],
    }

    out = {}
    for case, members in cases.items():
        mix = equal_weight_case_mixture(selected, members, FEATURE_COLS)
        for j, col in enumerate(FEATURE_COLS):
            mix[col] = (mix[col].to_numpy(dtype=float) - ref.mean[j]) / ref.std[j]
        out[case] = mix
    return out


def select_region_df(
    plane: PlaneData,
    s_min: float,
    s_max: float,
    z_agl_max: float,
    center_on_crest: bool,
    z_agl_min: float = -1.0,
) -> pd.DataFrame:
    return select_plane_window(
        plane=plane,
        s_min=s_min,
        s_max=s_max,
        z_agl_max=z_agl_max,
        center_on_crest=center_on_crest,
        z_agl_min=z_agl_min,
    )


def compute_regional_ood_summary(
    planes: Mapping[str, PlaneData],
    ref: H42Reference,
) -> pd.DataFrame:
    """
    Region-level OOD summary for the extra Fig.17(f,g)-style figure.
    Askervein and Bolund field transects are combined with equal transect weight.
    """
    rows = []

    # Askervein: combine A and AA equally for each region.
    for region, (s0, s1) in ASK_REGION_BOUNDS.items():
        parts = []
        per_member = []
        for key in ["ASK_A", "ASK_AA"]:
            sub = select_region_df(
                planes[key], s0, s1, ASK_ZAGL_MAX_M, center_on_crest=True
            )
            if len(sub) == 0:
                continue
            w, wmean = normalized_wasserstein(ref, sub)
            per_member.append({
                "member": key,
                "N": len(sub),
                "W_OOD_mean": wmean,
                **{f"W_{k}": float(v) for k, v in zip(FEATURE_KEYS, w)},
            })
            parts.append(sub)

        if per_member:
            row = {
                "case": "Askervein",
                "region": region,
                "n_members": len(per_member),
                "N_total": int(sum(x["N"] for x in per_member)),
                "F_OOD": float(np.mean([x["W_OOD_mean"] for x in per_member])),
            }
            for k in FEATURE_KEYS:
                row[f"W_{k}"] = float(np.mean([x[f"W_{k}"] for x in per_member]))
            rows.append(row)

    # Bolund 270 and 239: combine line A/B equally for each region.
    for case, members in [("Bolund 270°", ["B270_A", "B270_B"]),
                          ("Bolund 239°", ["B239_A", "B239_B"])]:
        for region, (s0, s1) in BOL_REGION_BOUNDS.items():
            per_member = []
            for key in members:
                sub = select_region_df(
                    planes[key], s0, s1, BOL_ZAGL_MAX_M, center_on_crest=False
                )
                if len(sub) == 0:
                    continue
                w, wmean = normalized_wasserstein(ref, sub)
                per_member.append({
                    "member": key,
                    "N": len(sub),
                    "W_OOD_mean": wmean,
                    **{f"W_{k}": float(v) for k, v in zip(FEATURE_KEYS, w)},
                })

            if per_member:
                row = {
                    "case": case,
                    "region": region,
                    "n_members": len(per_member),
                    "N_total": int(sum(x["N"] for x in per_member)),
                    "F_OOD": float(np.mean([x["W_OOD_mean"] for x in per_member])),
                }
                for k in FEATURE_KEYS:
                    row[f"W_{k}"] = float(np.mean([x[f"W_{k}"] for x in per_member]))
                rows.append(row)

    return pd.DataFrame(rows)


def plot_standardized_feature_distributions(
    std_cases: Mapping[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Additional figure: five standardized feature distributions,
    H42 vs H38/H31/Askervein/Bolund270/Bolund239.
    """
    configure_matplotlib()

    case_order = ["H42", "H38", "H31", "Askervein", "Bolund 270°", "Bolund 239°"]
    colors = {
        "H42": "#4D4D4D",
        "H38": "#2C7FB8",
        "H31": "#41B6C4",
        "Askervein": "#1f77b4",
        "Bolund 270°": "#ff7f0e",
        "Bolund 239°": "#9467bd",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.4))
    axes = axes.ravel()

    # Five feature panels + one legend/text panel.
    for j, (fkey, col, label) in enumerate(FEATURES):
        ax = axes[j]
        all_x = []

        for case in case_order:
            vals = std_cases[case][col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            if len(vals) > 4000:
                # Quantile-representative decimation for speed/stability.
                q = np.linspace(0.001, 0.999, 4000)
                vals_plot = np.quantile(vals, q)
            else:
                vals_plot = np.sort(vals)

            all_x.extend([np.nanmin(vals_plot), np.nanmax(vals_plot)])

            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals_plot)
            xgrid = np.linspace(np.min(vals_plot), np.max(vals_plot), 450)
            ygrid = kde(xgrid)
            ax.plot(xgrid, ygrid, lw=2.0, color=colors[case], label=case)

        ax.axvline(0.0, color="0.55", ls="--", lw=1.0)
        ax.set_title(f"{label}", pad=6.0)
        ax.set_xlabel("Standardized value relative to H42")
        if j in [0, 3]:
            ax.set_ylabel("Density")
        ax.grid(alpha=0.14, linewidth=0.5, color="0.75")

        if all_x:
            xmin = min(all_x)
            xmax = max(all_x)
            span = xmax - xmin
            ax.set_xlim(xmin - 0.06 * span, xmax + 0.06 * span)

    # Sixth panel: legend and short explanation.
    ax = axes[5]
    ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colors[c], lw=2.4, label=c)
        for c in case_order
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=11.0,
    )
    expl = (
        "Standardization uses the H42 mean and standard deviation.\n"
        "Askervein and Bolund curves are equal-transect mixtures\n"
        "to avoid bias from unequal Fluent cell counts."
    )
    ax.text(0.02, 0.40, expl, ha="left", va="top", fontsize=11.2)

    fig.suptitle(
        "Fig. 17(a–e). Standardized distributions of the five retained features",
        y=0.99, fontsize=16.0, fontweight="bold"
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09,
                        wspace=0.28, hspace=0.34)

    for ext in ["png", "pdf", "svg"]:
        fp = out_dir / f"Fig17a_e_standardized_feature_distributions.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fp, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regional_ood_summary_and_ml_scatter(
    regional_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Additional figure:
      (f) regional OOD score bar chart
      (g) regional OOD score vs GEKO-ML incremental U-performance
    """
    configure_matplotlib()

    # Order bars in a physically readable sequence.
    desired_order = [
        ("Askervein", "Windward"),
        ("Askervein", "Crest"),
        ("Askervein", "Leeward slope"),
        ("Bolund 270°", "Pre-cliff"),
        ("Bolund 270°", "Immediate post-cliff"),
        ("Bolund 270°", "Central plateau"),
        ("Bolund 270°", "Lee-side"),
        ("Bolund 239°", "Pre-cliff"),
        ("Bolund 239°", "Immediate post-cliff"),
        ("Bolund 239°", "Central plateau"),
        ("Bolund 239°", "Lee-side"),
    ]

    # Keep only rows that actually exist.
    keep = []
    for case, region in desired_order:
        rr = regional_df[(regional_df["case"] == case) & (regional_df["region"] == region)]
        if not rr.empty:
            keep.append(rr.iloc[0])
    reg = pd.DataFrame(keep).reset_index(drop=True)

    reg["label"] = reg["case"].str.replace("Bolund ", "B", regex=False) + "\n" + reg["region"]

    case_colors = {
        "Askervein": "#1f77b4",
        "Bolund 270°": "#ff7f0e",
        "Bolund 239°": "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.9))
    axf, axg = axes

    # ------------------------------ panel (f)
    x = np.arange(len(reg))
    colors = [case_colors.get(c, "#4D4D4D") for c in reg["case"]]
    bars = axf.bar(x, reg["F_OOD"], color=colors, edgecolor="black", linewidth=0.8)

    for b, v in zip(bars, reg["F_OOD"]):
        axf.text(
            b.get_x() + b.get_width() / 2, v + 0.015,
            f"{v:.2f}", ha="center", va="bottom",
            fontsize=9.3, rotation=90 if len(reg) > 8 else 0
        )

    axf.set_xticks(x)
    axf.set_xticklabels(reg["label"], rotation=45, ha="right")
    axf.set_ylabel(r"Regional OOD score, $F_{\mathrm{OOD}}$")
    axf.set_title("(f) Regional feature-space OOD score", loc="left", fontweight="bold")
    axf.grid(axis="y", alpha=0.16, linewidth=0.6, color="0.75")

    # ------------------------------ panel (g)
    plot_rows = []
    for _, r in reg.iterrows():
        key = (r["case"], r["region"])
        if key in REGIONAL_ML_GAIN_U:
            plot_rows.append({
                "case": r["case"],
                "region": r["region"],
                "F_OOD": r["F_OOD"],
                "G_ML_U": REGIONAL_ML_GAIN_U[key],
            })
    sc = pd.DataFrame(plot_rows)

    marker_map = {
        "Askervein": "o",
        "Bolund 270°": "s",
        "Bolund 239°": "D",
    }

    for case in ["Askervein", "Bolund 270°", "Bolund 239°"]:
        rr = sc[sc["case"] == case]
        if rr.empty:
            continue
        axg.scatter(
            rr["F_OOD"], rr["G_ML_U"],
            s=78, marker=marker_map[case],
            color=case_colors[case], edgecolors="black", linewidths=0.7,
            alpha=0.95, label=case
        )
        for _, r in rr.iterrows():
            axg.annotate(
                r["region"], (r["F_OOD"], r["G_ML_U"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9.0, color=case_colors[case]
            )

    axg.axhline(0.0, color="0.35", ls="--", lw=1.0)
    axg.set_xlabel(r"Regional OOD score, $D_{\mathrm{OOD}}$")
    axg.set_ylabel(r"Incremental GEKO-ML effect on $U$, $\Delta MSE_{\mathrm{ML}}$ (%)")
    axg.set_title("(g) Regional OOD vs GEKO-ML incremental effect", loc="left", fontweight="bold")
    axg.grid(alpha=0.16, linewidth=0.6, color="0.75")
    axg.legend(frameon=False, loc="best")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.22, wspace=0.22)

    for ext in ["png", "pdf", "svg"]:
        fp = out_dir / f"Fig17f_g_regional_OOD_summary.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fp, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# =============================================================================
# 11. PLOTTING
# =============================================================================


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 13.0,
        "axes.labelsize": 14.5,
        "axes.titlesize": 15.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_a_case_shift(ax: plt.Axes, case_df: pd.DataFrame) -> None:
    order = ["H42", "H38", "H31", "Askervein", "Bolund 270°", "Bolund 239°"]
    vals = [
        float(case_df.loc[case_df["case"] == c, "W_OOD_mean"].iloc[0])
        for c in order
    ]
    x = np.arange(len(order))

    colors = ["#D9D9D9", "#86BBD8", "#3D8CC2", "#F6AE2D", "#F26419", "#C44536"]

    bars = ax.bar(
        x, vals, width=0.72, color=colors,
        edgecolor="black", linewidth=0.9
    )

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=11.5, fontweight="bold"
        )

    ax.axvline(2.5, color="0.45", ls="--", lw=1.1)

    ymax = max(vals + [0.8])
    ax.text(1.0, ymax * 1.09, "In-family transfer",
            ha="center", va="bottom", fontsize=12.0, fontweight="bold")
    ax.text(4.0, ymax * 1.09, "Atmospheric-scale transfer",
            ha="center", va="bottom", fontsize=12.0, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["H42", "H38", "H31", "Askervein", "Bolund\n270°", "Bolund\n239°"]
    )
    ax.set_ylabel(r"Normalized distribution shift, $W_{\mathrm{OOD}}^*$")
    ax.set_ylim(0.0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7, color="0.7")
    ax.set_title(
        "(a) Case-level feature-space distribution shift",
        loc="left", fontweight="bold"
    )


ROW_ORDER = [
    ("H38", "H38"),
    ("H31", "H31"),
    ("ASK-A", "ASK_A"),
    ("ASK-AA", "ASK_AA"),
    ("B270-A", "B270_A"),
    ("B270-B", "B270_B"),
    ("B239-A", "B239_A"),
    ("B239-B", "B239_B"),
]

def matrix_from_summary(
    transect_df: pd.DataFrame,
    metric: str,
) -> Tuple[np.ndarray, List[str]]:
    mat = np.full((len(ROW_ORDER), len(FEATURE_KEYS)), np.nan)
    labels = []

    for i, (display, dataset) in enumerate(ROW_ORDER):
        labels.append(display)
        r = transect_df[transect_df["dataset"] == dataset]
        if r.empty:
            continue
        r = r.iloc[0]
        for j, key in enumerate(FEATURE_KEYS):
            if metric == "exceedance":
                col = f"exceed_{key}_pct"
            elif metric == "wasserstein":
                col = f"W_{key}"
            else:
                raise ValueError(f"Unknown heatmap metric: {metric}")
            mat[i, j] = float(r[col])

    return mat, labels



def add_heatmap_numbers(
    ax: plt.Axes,
    mat: np.ndarray,
    mode: str,
    threshold: float,
) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "–"
            elif mode == "percent":
                txt = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            else:
                txt = f"{v:.2f}"

            color = "white" if np.isfinite(v) and v > threshold else "black"
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=10.2, color=color, fontweight="bold"
            )


def panel_b_exceedance(ax: plt.Axes, transect_df: pd.DataFrame) -> None:
    mat, labels = matrix_from_summary(transect_df, "exceedance")

    im = ax.imshow(
        mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100,
        interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(FEATURE_KEYS)))
    ax.set_xticklabels(FEATURE_LABELS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    add_heatmap_numbers(ax, mat, mode="percent", threshold=55.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.040)
    cbar.set_label("Outside H42 0.5–99.5% support (%)", fontsize=12.0)
    cbar.ax.tick_params(labelsize=11.0)

    ax.set_title(
        "(b) Feature-wise extrapolation beyond H42 support",
        loc="left", fontweight="bold"
    )


def panel_c_wasserstein_heatmap(
    ax: plt.Axes,
    transect_df: pd.DataFrame,
) -> None:
    mat, labels = matrix_from_summary(transect_df, "wasserstein")

    vmax = max(1.0, float(np.nanpercentile(mat, 98)))
    im = ax.imshow(
        mat, aspect="auto", cmap="YlGnBu",
        vmin=0.0, vmax=vmax, interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(FEATURE_KEYS)))
    ax.set_xticklabels(FEATURE_LABELS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    add_heatmap_numbers(
        ax, mat, mode="number",
        threshold=0.52 * vmax
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.040)
    cbar.set_label(r"Normalized Wasserstein shift, $W_j^*$", fontsize=12.0)
    cbar.ax.tick_params(labelsize=11.0)

    ax.set_title(
        "(c) Feature-wise distribution shift",
        loc="left", fontweight="bold"
    )

def sample_rows(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)].copy()


def equal_sample_case(
    selected: Mapping[str, pd.DataFrame],
    members: Sequence[str],
    n_each: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    parts = [sample_rows(selected[m], n_each, rng) for m in members]
    return pd.concat(parts, ignore_index=True)




def panel_d_phi7_phi8(
    ax: plt.Axes,
    selected: Mapping[str, pd.DataFrame],
    ref: H42Reference,
) -> None:
    """
    Publication-oriented phi7-phi8 diagnostic.

    IMPORTANT:
    - The light background is the H42 *joint 2-D density* only.
    - The dashed rectangle is the Cartesian product of the two marginal
      H42 Q0.5%-Q99.5% intervals. Therefore it is labelled "marginal
      robust envelope", not "joint 99% support".
    - Target cases are summarized by the median and marginal 5%-95%
      ranges. This avoids non-comparable KDE contours normalized to each
      case's own peak density.
    - Atmospheric field transects are equally weighted when their case-
      level marginal quantiles are formed.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib.lines import Line2D

    xcol = "turbulent-reynolds-number-scaled"
    ycol = "turbulent-viscosity-ratio-scaled"

    # ------------------------------------------------------------------
    # A) H42 joint density background using a smoothed 2-D histogram.
    #    Faster and more stable than a full 118k-point Gaussian KDE.
    # ------------------------------------------------------------------
    xh = selected["H42"][xcol].to_numpy(dtype=float)
    yh = selected["H42"][ycol].to_numpy(dtype=float)

    finite = np.isfinite(xh) & np.isfinite(yh)
    xh = xh[finite]
    yh = yh[finite]

    # Use the full physical range while adding a small plotting margin.
    xmin = min(-0.02, float(np.nanmin(xh)) - 0.02)
    xmax = max(2.03, float(np.nanmax(xh)) + 0.02)
    ymin = min(-0.02, float(np.nanmin(yh)) - 0.02)
    ymax = max(2.03, float(np.nanmax(yh)) + 0.02)

    nbins = 180
    H, xedges, yedges = np.histogram2d(
        xh, yh,
        bins=nbins,
        range=[[xmin, xmax], [ymin, ymax]],
        density=True,
    )
    H = gaussian_filter(H, sigma=1.25)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")

    positive = H[H > 0]
    if len(positive):
        # Use density quantiles, not arbitrary fractions of each case's peak.
        lev = np.quantile(positive, [0.45, 0.68, 0.84, 0.94])
        lev = np.unique(lev)
        if len(lev) >= 2:
            ax.contourf(
                XX, YY, H,
                levels=np.r_[lev, H.max() * 1.001],
                cmap="Greys",
                alpha=0.34,
                antialiased=True,
            )
            ax.contour(
                XX, YY, H,
                levels=lev,
                colors="0.50",
                linewidths=0.55,
                alpha=0.65,
            )

    # ------------------------------------------------------------------
    # B) H42 marginal robust envelope used consistently with panel (b).
    # ------------------------------------------------------------------
    j7 = FEATURE_KEYS.index("phi7")
    j8 = FEATURE_KEYS.index("phi8")
    x0, x1 = ref.q_low[j7], ref.q_high[j7]
    y0, y1 = ref.q_low[j8], ref.q_high[j8]

    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False,
        linestyle="--",
        linewidth=1.6,
        edgecolor="black",
        zorder=6,
    )
    ax.add_patch(rect)

    ax.text(
        x0 + 0.025 * (x1 - x0),
        y1 - 0.035 * max(y1 - y0, 1e-8),
        "H42 marginal 0.5–99.5% envelope",
        ha="left",
        va="top",
        fontsize=10.0,
        color="black",
        zorder=8,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.86,
            pad=1.4,
        ),
    )

    # ------------------------------------------------------------------
    # C) Equal-transect marginal mixture quantiles.
    # ------------------------------------------------------------------
    def equal_weight_marginal_quantiles(
        members: Sequence[str],
        col: str,
        probs: Sequence[float] = (0.05, 0.50, 0.95),
        nq: int = 12000,
    ) -> np.ndarray:
        """
        Construct an equal-weight empirical mixture without allowing the
        denser Fluent plane to dominate. Each transect contributes the
        same number of quantile-representative samples.
        """
        qgrid = (np.arange(nq, dtype=float) + 0.5) / nq
        parts = []
        for member in members:
            vals = selected[member][col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            parts.append(np.quantile(vals, qgrid))
        if not parts:
            return np.full(len(probs), np.nan)
        mix = np.concatenate(parts)
        return np.quantile(mix, probs)

    cases = [
        {
            "label": "H38",
            "members": ["H38"],
            "marker": "o",
            "color": "#2166AC",
            "ms": 7.0,
            "zorder": 10,
        },
        {
            "label": "H31",
            "members": ["H31"],
            "marker": "^",
            "color": "#1B9E77",
            "ms": 7.5,
            "zorder": 11,
        },
        {
            "label": "Askervein",
            "members": ["ASK_A", "ASK_AA"],
            "marker": "o",
            "color": "#E69F00",
            "ms": 9.0,
            "zorder": 14,
        },
        {
            "label": "Bolund 270°",
            "members": ["B270_A", "B270_B"],
            "marker": "s",
            "color": "#D55E00",
            "ms": 9.0,
            "zorder": 15,
        },
        {
            "label": "Bolund 239°",
            "members": ["B239_A", "B239_B"],
            "marker": "D",
            "color": "#7B3294",
            "ms": 8.5,
            "zorder": 16,
        },
    ]

    summary_rows = []

    for case in cases:
        qx = equal_weight_marginal_quantiles(case["members"], xcol)
        qy = equal_weight_marginal_quantiles(case["members"], ycol)

        if not (np.all(np.isfinite(qx)) and np.all(np.isfinite(qy))):
            continue

        xlo, xmed, xhi = qx
        ylo, ymed, yhi = qy

        summary_rows.append({
            "case": case["label"],
            "phi7_p05": xlo,
            "phi7_median": xmed,
            "phi7_p95": xhi,
            "phi8_p05": ylo,
            "phi8_median": ymed,
            "phi8_p95": yhi,
        })

        # Marginal 5%-95% ranges.
        ax.plot(
            [xlo, xhi], [ymed, ymed],
            color=case["color"], lw=1.7, alpha=0.95,
            zorder=case["zorder"] - 1,
        )
        ax.plot(
            [xmed, xmed], [ylo, yhi],
            color=case["color"], lw=1.7, alpha=0.95,
            zorder=case["zorder"] - 1,
        )

        # Hollow markers keep overlapping atmospheric points visible.
        ax.plot(
            xmed, ymed,
            marker=case["marker"],
            markersize=case["ms"],
            markerfacecolor="white",
            markeredgecolor=case["color"],
            markeredgewidth=2.0,
            linestyle="none",
            zorder=case["zorder"],
        )

    # ------------------------------------------------------------------
    # D) Clear text callouts for the near-(2,2) atmospheric cluster.
    #    These are annotations only; data coordinates are not shifted.
    # ------------------------------------------------------------------
    if summary_rows:
        ssum = pd.DataFrame(summary_rows).set_index("case")

        offsets = {
            "H38": (-56, 15),
            "H31": (-48, -23),
            "Askervein": (-106, 38),
            "Bolund 270°": (-111, 12),
            "Bolund 239°": (-110, -17),
        }

        for case in cases:
            label = case["label"]
            if label not in ssum.index:
                continue
            xmed = float(ssum.loc[label, "phi7_median"])
            ymed = float(ssum.loc[label, "phi8_median"])
            dx, dy = offsets[label]

            ax.annotate(
                label,
                xy=(xmed, ymed),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.0,
                color=case["color"],
                fontweight="bold",
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=case["color"],
                    lw=0.9,
                    shrinkA=2,
                    shrinkB=2,
                ),
                zorder=20,
            )

    # ------------------------------------------------------------------
    # E) Styling.
    # ------------------------------------------------------------------
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\phi_7$: scaled turbulent Reynolds number")
    ax.set_ylabel(r"$\phi_8$: scaled turbulent viscosity ratio")
    ax.grid(alpha=0.15, linewidth=0.6, color="0.75")

    legend_handles = [
        Line2D(
            [0], [0],
            marker=c["marker"],
            markersize=c["ms"] * 0.82,
            markerfacecolor="white",
            markeredgecolor=c["color"],
            markeredgewidth=1.7,
            linestyle="-",
            linewidth=1.4,
            color=c["color"],
            label=c["label"],
        )
        for c in cases
    ]

    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="0.80",
        loc="upper left",
        ncol=2,
        fontsize=8.7,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=0.9,
        borderpad=0.6,
    )

    ax.set_title(
        r"(d) $\phi_7$–$\phi_8$ transfer relative to the H42 reference",
        loc="left",
        fontweight="bold",
    )

def make_main_figure(
    case_df: pd.DataFrame,
    transect_df: pd.DataFrame,
    selected: Mapping[str, pd.DataFrame],
    ref: H42Reference,
    out_dir: Path,
) -> None:
    configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    axa, axb, axc, axd = axes.ravel()

    panel_a_case_shift(axa, case_df)
    panel_b_exceedance(axb, transect_df)
    panel_c_wasserstein_heatmap(axc, transect_df)
    panel_d_phi7_phi8(axd, selected, ref)

    fig.subplots_adjust(
        left=0.07, right=0.97,
        top=0.96, bottom=0.08,
        wspace=0.26, hspace=0.28
    )

    for ext in ["png", "pdf", "svg"]:
        fn = out_dir / f"Fig17_OOD_feature_space_transferability.{ext}"
        if ext == "png":
            fig.savefig(fn, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fn, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {fn}")

    plt.close(fig)

def plot_mahalanobis_diagnostic(
    transect_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    configure_matplotlib()

    order = [
        "H42", "H38", "H31",
        "ASK_A", "ASK_AA",
        "B270_A", "B270_B",
        "B239_A", "B239_B",
    ]
    labels = [
        "H42", "H38", "H31",
        "ASK-A", "ASK-AA",
        "B270-A", "B270-B",
        "B239-A", "B239-B",
    ]

    d = transect_df.set_index("dataset").reindex(order)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(order))

    ax.plot(x, d["median_Dstar"], marker="o", lw=1.2, label="Median")
    ax.plot(x, d["P90_Dstar"], marker="s", lw=1.2, label="P90")
    ax.plot(x, d["P99_Dstar"], marker="^", lw=1.2, label="P99")
    ax.axhline(
        1.0, color="0.35", ls="--", lw=1.0,
        label="H42 P99 normalization"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"Percentile-normalized $D_M^*$")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)

    ax.set_title(
        "Supplementary diagnostic: covariance-based Mahalanobis distance"
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "FigS_Mahalanobis_diagnostic.png",
        dpi=SAVE_DPI, bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "FigS_Mahalanobis_diagnostic.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_feature_cdfs(
    selected: Mapping[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    configure_matplotlib()

    groups = [
        ("H42", selected["H42"], "black"),
        ("H38", selected["H38"], "#4C78A8"),
        ("H31", selected["H31"], "#72B7B2"),
        ("ASK-A", selected["ASK_A"], "#F2A541"),
        ("ASK-AA", selected["ASK_AA"], "#D8902F"),
        ("B270-A", selected["B270_A"], "#E45756"),
        ("B270-B", selected["B270_B"], "#C93F3E"),
        ("B239-A", selected["B239_A"], "#8E5EA2"),
        ("B239-B", selected["B239_B"], "#6F4680"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15.8, 3.2))

    for j, (_, col, label) in enumerate(FEATURES):
        ax = axes[j]

        for name, df, color in groups:
            vals = np.sort(df[col].to_numpy(dtype=float))
            if len(vals) > 5000:
                q = np.linspace(0.0, 1.0, 5000)
                xx = np.quantile(vals, q)
                yy = q
            else:
                xx = vals
                yy = np.linspace(0.0, 1.0, len(vals))

            ax.plot(xx, yy, lw=0.95, color=color, label=name)

        ax.set_xlabel(label)
        if j == 0:
            ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.16, linewidth=0.5)

    axes[0].legend(frameon=False, fontsize=6.7)
    fig.tight_layout()

    fig.savefig(
        out_dir / "FigS_feature_CDFs.png",
        dpi=SAVE_DPI, bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "FigS_feature_CDFs.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# 13. TEXT SUMMARY
# =============================================================================

def write_text_summary(
    path: Path,
    case_df: pd.DataFrame,
    transect_df: pd.DataFrame,
    ref: H42Reference,
    planes: Mapping[str, PlaneData],
) -> None:
    lines = []
    lines.append("SECTION 6.4 OOD ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Case-level normalized distribution shift W*_OOD:")
    for _, r in case_df.iterrows():
        lines.append(f"  {r['case']:<15s}: {r['W_OOD_mean']:.6f}")
    lines.append("")

    lines.append("Feature-wise H42 robust support:")
    for j, key in enumerate(FEATURE_KEYS):
        lines.append(
            f"  {key}: [{ref.q_low[j]:.9g}, {ref.q_high[j]:.9g}]"
        )
    lines.append("")

    lines.append("Field feature exceedance (%):")
    for ds in ["ASK_A", "ASK_AA", "B270_A", "B270_B", "B239_A", "B239_B"]:
        r = transect_df[transect_df["dataset"] == ds].iloc[0]
        vals = ", ".join(
            f"{k}={r[f'exceed_{k}_pct']:.2f}%"
            for k in FEATURE_KEYS
        )
        lines.append(f"  {ds}: {vals}")
    lines.append("")

    lines.append("Askervein automatically estimated crest positions in initial s coordinate:")
    for ds in ["ASK_A", "ASK_AA"]:
        lines.append(f"  {ds}: s_crest={planes[ds].crest_s_raw:.3f} m")
    lines.append("")

    lines.append("Interpretation guardrail:")
    lines.append(
        "  W*_OOD and support exceedance quantify feature-space distribution shift/"
        "extrapolative deployment. They are not deterministic predictors of model error."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 14. COMMAND LINE
# =============================================================================



def phi7_phi8_case_summary(
    selected: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Case-level equal-transect phi7/phi8 marginal 5/50/95% statistics."""
    xcol = "turbulent-reynolds-number-scaled"
    ycol = "turbulent-viscosity-ratio-scaled"

    groups = {
        "H42": ["H42"],
        "H38": ["H38"],
        "H31": ["H31"],
        "Askervein": ["ASK_A", "ASK_AA"],
        "Bolund 270°": ["B270_A", "B270_B"],
        "Bolund 239°": ["B239_A", "B239_B"],
    }

    qgrid = (np.arange(20000, dtype=float) + 0.5) / 20000.0
    rows = []

    def equal_mix_quantiles(members, col):
        parts = []
        for m in members:
            vals = selected[m][col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            parts.append(np.quantile(vals, qgrid))
        if not parts:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        mix = np.concatenate(parts)
        return np.quantile(mix, [0.05, 0.50, 0.95])

    for case, members in groups.items():
        qx = equal_mix_quantiles(members, xcol)
        qy = equal_mix_quantiles(members, ycol)
        rows.append({
            "case": case,
            "phi7_p05": qx[0],
            "phi7_median": qx[1],
            "phi7_p95": qx[2],
            "phi8_p05": qy[0],
            "phi8_median": qy[1],
            "phi8_p95": qy[2],
        })

    return pd.DataFrame(rows)



# =============================================================================
# FINAL FIG.17 — 2 x 3 spatial / regional / transferability figure
# =============================================================================


def configure_final_fig_style() -> None:
    """Large, publication-ready fonts for the final Sec. 6.4 Fig.17."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 12.5,
        "axes.labelsize": 13.6,
        "axes.titlesize": 14.2,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "legend.fontsize": 9.6,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def h42_mahalanobis_reference_sorted(ref: H42Reference) -> np.ndarray:
    """Sorted H42 Mahalanobis-distance reference distribution."""
    Z = (ref.values - ref.mean[None, :]) / ref.std[None, :]
    d = mahalanobis_values_from_z(Z, ref.cov_z_inv)
    return np.sort(d)


def pointwise_raw_mahalanobis(
    ref: H42Reference,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Raw covariance-aware Mahalanobis distance in the retained five-feature space.

        D_M(x) = sqrt[(z-mu_H42)^T Sigma_H42^{-1} (z-mu_H42)]

    Here z denotes the H42-standardized feature vector. Standardizing first and
    then using the covariance of standardized H42 features is algebraically
    equivalent to using the raw covariance matrix, while improving numerical
    conditioning.
    """
    X = feature_matrix(df)
    Z = (X - ref.mean[None, :]) / ref.std[None, :]
    return mahalanobis_values_from_z(Z, ref.cov_z_inv)


def percentile_normalize_mahalanobis(
    raw_d: np.ndarray,
    h42_d_sorted: np.ndarray,
) -> np.ndarray:
    r"""
    Percentile-normalized Mahalanobis distance following the extrapolation
    normalization of Wu et al. (Flow, Turbulence and Combustion, 2017):

        D_M^*(x) = 1 - gamma_dm
                 = F_H42[D_M(x)]

    where gamma_dm is the fraction of H42-reference samples having a larger
    raw Mahalanobis distance than the target point. Thus D_M^* lies in [0,1]
    and is the empirical H42 radial percentile of the target feature vector.

    This is an applicability / feature-space support score, NOT a calibrated
    probability of prediction error.
    """
    raw_d = np.asarray(raw_d, dtype=float)
    h42_d_sorted = np.asarray(h42_d_sorted, dtype=float)
    if len(h42_d_sorted) == 0:
        return np.full_like(raw_d, np.nan, dtype=float)

    ranks = np.searchsorted(h42_d_sorted, raw_d, side="right")
    return ranks.astype(float) / float(len(h42_d_sorted))


def pointwise_normalized_mahalanobis(
    ref: H42Reference,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Literature-consistent percentile-normalized Mahalanobis distance used in
    the FINAL Fig.17.
    """
    raw_d = pointwise_raw_mahalanobis(ref, df)
    h42_sorted = h42_mahalanobis_reference_sorted(ref)
    return percentile_normalize_mahalanobis(raw_d, h42_sorted)


def pointwise_q99_scaled_mahalanobis(
    ref: H42Reference,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Secondary threshold-scaled diagnostic retained for reproducibility:

        D_M,Q99 = D_M / Q99[D_M(H42)]

    D_M,Q99 = 1 corresponds to the H42 empirical 99th percentile. This score is
    NOT the percentile normalization used in the main figure.
    """
    return pointwise_raw_mahalanobis(ref, df) / float(ref.mahal_p99)


def _equal_weight_quantile_mix(arrays, probs, n_quantile_samples=20000):
    qgrid = (np.arange(n_quantile_samples, dtype=float) + 0.5) / n_quantile_samples
    parts = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a):
            parts.append(np.quantile(a, qgrid))
    if not parts:
        return np.full(len(probs), np.nan)
    return np.quantile(np.concatenate(parts), probs)





def build_askervein_validation_targets_hardcoded():
    """Build Askervein validation targets without reading OBS.xlsx.

    U   : A1-A9 + AA10-AA24 = 24 locations at z_AGL=10 m.
    TKE : A1-A9 + 3 sparse AA TKE locations = 12 locations at z_AGL=10 m.

    Sampling coordinates:
      ASK_A  -> s_raw_m referenced to HT.
      ASK_AA -> s_raw_m referenced to CP.
    """
    rows = []

    # Line A: same 9 physical positions for U and TKE.
    for point, mast_name, s_m in ASK_LINE_A_VALIDATION:
        region = ASK_REGION_FROM_POINT[point]
        for variable in ["U", "TKE"]:
            rows.append({
                "case": "Askervein",
                "variable": variable,
                "point": point,
                "mast": mast_name,
                "plane_key": "ASK_A",
                "s_target_m": float(s_m),
                "z_agl_m": ASK_VALIDATION_Z_AGL_M,
                "region": region,
                "physical_id": f"ASK_A_{mast_name}_z10",
                "source_support": "hard-coded Askervein Line A validation location",
            })

    # Line AA U: all 15 U locations.
    for point, s_m in ASK_LINE_AA_U_VALIDATION:
        rows.append({
            "case": "Askervein",
            "variable": "U",
            "point": point,
            "mast": point,
            "plane_key": "ASK_AA",
            "s_target_m": float(s_m),
            "z_agl_m": ASK_VALIDATION_Z_AGL_M,
            "region": ASK_REGION_FROM_POINT[point],
            "physical_id": f"ASK_AA_{point}_z10",
            "source_support": "hard-coded Askervein Line AA U validation location",
        })

    # Line AA TKE: only 3 actual sparse TKE mast positions.
    for point, mast_name, s_m in ASK_LINE_AA_TKE_VALIDATION:
        rows.append({
            "case": "Askervein",
            "variable": "TKE",
            "point": point,
            "mast": mast_name,
            "plane_key": "ASK_AA",
            "s_target_m": float(s_m),
            "z_agl_m": ASK_VALIDATION_Z_AGL_M,
            "region": "Windward",
            "physical_id": f"ASK_AA_{mast_name}_z10",
            "source_support": "hard-coded Askervein Line AA sparse TKE mast",
        })

    out = pd.DataFrame(rows)

    # Strict audit against the manuscript's regional sample counts.
    expected = {
        ("U", "Windward"): 12,
        ("U", "Crest"): 5,
        ("U", "Lee"): 7,
        ("TKE", "Windward"): 8,
        ("TKE", "Crest"): 1,
        ("TKE", "Lee"): 3,
    }
    got = out.groupby(["variable", "region"]).size().to_dict()
    for key, n_exp in expected.items():
        n_got = int(got.get(key, 0))
        if n_got != n_exp:
            raise RuntimeError(
                f"Askervein hard-coded support mismatch for {key}: "
                f"got {n_got}, expected {n_exp}"
            )

    print("[ASK VALIDATION SUPPORT: HARD-CODED; NO OBS FILE]")
    print(out.groupby(["variable", "region"]).size().rename("N").reset_index().to_string(index=False))
    print(
        f"[ASK REF] HT={REFERENCE_POINTS_XY['ASK_A']}, "
        f"CP={REFERENCE_POINTS_XY['ASK_AA']}, "
        f"z_AGL={ASK_VALIDATION_Z_AGL_M:.1f} m"
    )
    return out


def build_bolund_validation_targets():
    """
    Build the Bolund validation support at the mast positions and measurement
    heights used by the manuscript regional statistics.

    M3 is sampled from BOTH Line A and Line B and then collapsed to one
    physical-location value per height by averaging the two extracted D_M*.
    """
    rows = []

    case_defs = [
        ("Bolund 270°", "B270_A", "B270_B", BOLUND_HEIGHTS_270),
        ("Bolund 239°", "B239_A", "B239_B", BOLUND_HEIGHTS_239),
    ]

    for case, key_a, key_b, heights_by_mast in case_defs:
        for variable in ["U", "TKE"]:
            for mast, heights in heights_by_mast.items():
                region = BOLUND_REGION_FROM_MAST[mast]

                # Determine which transect(s) contain this physical mast.
                if mast in BOLUND_MAST_S_BY_LINE["A"]:
                    planes_for_mast = [
                        (key_a, BOLUND_MAST_S_BY_LINE["A"][mast], "Line A")
                    ]
                else:
                    planes_for_mast = []

                if mast in BOLUND_MAST_S_BY_LINE["B"]:
                    planes_for_mast.append(
                        (key_b, BOLUND_MAST_S_BY_LINE["B"][mast], "Line B")
                    )

                for z_agl in heights:
                    phys = f"{case}_{mast}_z{z_agl:g}"
                    for plane_key, s_target, line_name in planes_for_mast:
                        rows.append({
                            "case": case,
                            "variable": variable,
                            "point": mast,
                            "mast": mast,
                            "plane_key": plane_key,
                            "s_target_m": float(s_target),
                            "z_agl_m": float(z_agl),
                            "region": region,
                            "physical_id": phys,
                            "source_support": f"Bolund {line_name} mast/instrument height",
                        })

    out = pd.DataFrame(rows)

    # Audit that the physical mast/height support exactly reproduces
    # the regional N used by Tables III/IV. U and TKE intentionally use
    # the same Bolund measurement support.
    for case, expected in EXPECTED_BOLUND_REGION_COUNTS.items():
        chk = (
            out[(out["case"] == case) & (out["variable"] == "U")]
            .groupby("region")["physical_id"]
            .nunique()
            .to_dict()
        )
        for region, n_expected in expected.items():
            n_got = int(chk.get(region, 0))
            if n_got != int(n_expected):
                raise RuntimeError(
                    f"Bolund validation-support mismatch: {case}, {region}: "
                    f"got N={n_got}, expected N={n_expected}"
                )
    return out


def build_validation_target_table():
    ask = build_askervein_validation_targets_hardcoded()
    bol = build_bolund_validation_targets()
    return pd.concat([ask, bol], ignore_index=True)


def _sample_one_plane_target_features(
    df,
    target_rows,
    s_col,
):
    """
    Interpolate the five retained NN input features to each validation
    location. Linear interpolation is used first; points outside the local
    triangulation fall back to nearest-neighbor interpolation.

    The Mahalanobis transform is deliberately applied AFTER feature
    interpolation because D_M(q) is nonlinear in q.
    """
    from scipy.interpolate import griddata

    if target_rows.empty:
        return np.empty((0, len(FEATURE_COLS))), np.array([], dtype=object)

    s = df[s_col].to_numpy(dtype=float)
    z = df["z_agl_m"].to_numpy(dtype=float)

    finite_geometry = np.isfinite(s) & np.isfinite(z)
    pts = np.column_stack([s[finite_geometry], z[finite_geometry]])
    tq = target_rows[["s_target_m", "z_agl_m"]].to_numpy(dtype=float)

    Xq = np.full((len(tq), len(FEATURE_COLS)), np.nan, dtype=float)
    method = np.full(len(tq), "linear", dtype=object)

    for j, col in enumerate(FEATURE_COLS):
        val = df[col].to_numpy(dtype=float)[finite_geometry]
        good = np.isfinite(val)
        q = griddata(pts[good], val[good], tq, method="linear")

        miss = ~np.isfinite(q)
        if np.any(miss):
            q_near = griddata(pts[good], val[good], tq[miss], method="nearest")
            q[miss] = q_near
            method[miss] = "nearest-fallback"

        Xq[:, j] = q

    return Xq, method


def _mahalanobis_from_feature_matrix(
    X,
    ref,
    h42_d_sorted,
):
    X = np.asarray(X, dtype=float)
    Z = (X - ref.mean[None, :]) / ref.std[None, :]
    raw_d = mahalanobis_values_from_z(Z, ref.cov_z_inv)

    # Main literature-consistent score.
    dstar = percentile_normalize_mahalanobis(raw_d, h42_d_sorted)

    # Secondary Q99-scaled diagnostic.
    dq99 = raw_d / float(ref.mahal_p99)

    return raw_d, dstar, dq99


def sample_validation_location_mahalanobis(selected_field, ref, targets):
    """
    Sample the retained five NN inputs at the actual validation locations,
    then compute Mahalanobis distance in feature space.

    Coordinate convention:
      ASK_A  -> s_raw_m  referenced to HT.
      ASK_AA -> s_raw_m  referenced to CP.
      Bolund -> s_m      referenced to CP.

    For Bolund M3, which is represented by both Line A and Line B, the
    interpolated FEATURE VECTORS from the two planes are averaged first; the
    nonlinear Mahalanobis transform is then applied once to that averaged
    physical-location feature vector.
    """
    sampled_parts = []
    h42_d_sorted = h42_mahalanobis_reference_sorted(ref)

    for plane_key, tg in targets.groupby("plane_key", sort=False):
        if plane_key not in selected_field:
            raise KeyError(f"Validation target references missing plane: {plane_key}")

        df = selected_field[plane_key]
        s_col = "s_raw_m" if plane_key in {"ASK_A", "ASK_AA"} else "s_m"

        Xq, methods = _sample_one_plane_target_features(
            df=df,
            target_rows=tg,
            s_col=s_col,
        )

        tmp = tg.copy()
        for j, col in enumerate(FEATURE_COLS):
            tmp[f"interp_{col}"] = Xq[:, j]

        raw_d, dstar, dq99 = _mahalanobis_from_feature_matrix(
            Xq, ref, h42_d_sorted
        )
        tmp["D_M_raw_plane"] = raw_d
        tmp["D_M_star_plane"] = dstar
        tmp["D_M_Q99_plane"] = dq99
        tmp["interp_method"] = methods
        tmp["sampling_s_column"] = s_col
        sampled_parts.append(tmp)

    raw = pd.concat(sampled_parts, ignore_index=True)

    # Collapse multiple plane representations of the same physical location by
    # averaging the interpolated FEATURE VECTOR first.
    group_cols = [
        "case", "variable", "point", "mast", "region",
        "physical_id", "z_agl_m",
    ]
    agg_spec = {
        f"interp_{col}": "mean" for col in FEATURE_COLS
    }
    agg_spec.update({
        "D_M_raw_plane": "mean",
        "D_M_star_plane": "mean",
        "D_M_Q99_plane": "mean",
        "source_support": lambda s: " | ".join(sorted(set(map(str, s)))),
        "interp_method": lambda s: " | ".join(sorted(set(map(str, s)))),
    })

    collapsed = (
        raw.groupby(group_cols, as_index=False, dropna=False)
           .agg(agg_spec)
    )

    # Number of contributing planes.
    nrep = (
        raw.groupby(group_cols, as_index=False, dropna=False)
           .size()
           .rename(columns={"size": "n_plane_representations"})
    )
    collapsed = collapsed.merge(nrep, on=group_cols, how="left")

    Xphys = collapsed[[f"interp_{c}" for c in FEATURE_COLS]].to_numpy(dtype=float)
    raw_d, dstar, dq99 = _mahalanobis_from_feature_matrix(
        Xphys, ref, h42_d_sorted
    )
    collapsed["D_M_raw"] = raw_d
    collapsed["D_M_star"] = dstar
    collapsed["D_M_Q99"] = dq99

    return raw, collapsed


def summarize_mast_location_mahalanobis(samples):
    """
    Region-level summary based ONLY on validation mast/measurement locations.

    For Askervein:
      U   -> A1-A9 + AA10-AA24 at 10 m AGL
      TKE -> only locations with available TKE observations.

    For Bolund:
      U/TKE -> mast/instrument-height support configured above.
    """
    rows = []
    if samples.empty:
        return pd.DataFrame()

    for (case, variable, region), g in samples.groupby(
        ["case", "variable", "region"], sort=False
    ):
        v = g["D_M_star"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue

        q05, q25, q50, q75, q95 = np.quantile(
            v, [0.05, 0.25, 0.50, 0.75, 0.95]
        )

        rows.append({
            "case": case,
            "variable": variable,
            "region": region,
            "N_validation_points": int(len(v)),
            "DMstar_Q05": float(q05),
            "DMstar_Q25": float(q25),
            "DMstar_median": float(q50),
            "DMstar_Q75": float(q75),
            "DMstar_Q95": float(q95),
            "DMstar_IQR": float(q75 - q25),
            "DMstar_mean": float(np.mean(v)),
            "DMstar_min": float(np.min(v)),
            "DMstar_max": float(np.max(v)),
        })

    out = pd.DataFrame(rows)

    print("[MAST-LOCATION D_M* SUPPORT]")
    if not out.empty:
        print(
            out[["case", "variable", "region", "N_validation_points", "DMstar_median"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    return out

def build_final_ml_gain_table():
    rows = []
    for (case, region, var), (r0, r1) in FINAL_ML_RMSE.items():
        # Absolute RMSE change, consistent with the RMSE values reported in
        # the preceding regional tables:
        #   DeltaRMSE_ML = RMSE_Std-GEKO - RMSE_GEKO-ML
        #
        # Positive: GEKO-ML reduces RMSE (improvement)
        # Negative: GEKO-ML increases RMSE (degradation)
        delta_rmse = r0 - r1
        rows.append({
            "case": case,
            "region": region,
            "variable": var,
            "RMSE_Std_GEKO": r0,
            "RMSE_GEKO_ML": r1,
            "DeltaRMSE_ML": delta_rmse,
        })
    return pd.DataFrame(rows)




def _shared_spatial_mahal_limits(selected_field, ref):
    """
    Shared colorbar limits for all six atmospheric transects.

    Uses pooled field-case D_M* quantiles only for visualization. This avoids
    compressing the entire map into a narrow mid-color band when most values
    are far below the H42-P99 normalization level (=1).
    """
    pooled = []
    for key in ["ASK_A", "ASK_AA", "B270_A", "B270_B", "B239_A", "B239_B"]:
        if key not in selected_field:
            continue
        v = pointwise_normalized_mahalanobis(ref, selected_field[key])
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            pooled.append(v)

    if not pooled:
        return 0.0, 1.0

    vals = np.concatenate(pooled)
    qlo, qhi = np.quantile(vals, FINAL_MAHAL_DISPLAY_QUANTILES)

    step = float(FINAL_MAHAL_DISPLAY_ROUND)
    vmin = np.floor(qlo / step) * step
    vmax = np.ceil(qhi / step) * step

    if vmax <= vmin:
        vmax = vmin + step

    print(
        f"[FINAL FIG17] shared spatial D_M* color range = "
        f"[{vmin:.3f}, {vmax:.3f}] "
        f"from pooled q{100*FINAL_MAHAL_DISPLAY_QUANTILES[0]:.0f}-"
        f"q{100*FINAL_MAHAL_DISPLAY_QUANTILES[1]:.0f}"
    )
    return float(vmin), float(vmax)

def _spatial_dood_grid(df, dstar, nbins_s, nbins_z):
    """Median normalized Mahalanobis distance D_M^* in regular (s,z) bins."""
    from scipy.stats import binned_statistic_2d

    s = df["s_m"].to_numpy(dtype=float)
    z = df["z-coordinate"].to_numpy(dtype=float)
    terrain = df["terrain_z_m"].to_numpy(dtype=float)

    zdatum = float(np.nanmin(terrain))
    zrel = z - zdatum

    stat, sedges, zedges, _ = binned_statistic_2d(
        s, zrel, dstar,
        statistic="median",
        bins=[nbins_s, nbins_z],
    )
    return stat, sedges, zedges, zdatum

def _terrain_profile_for_selected_df(df, zdatum, nbins=350):
    s = df["s_m"].to_numpy(dtype=float)
    tz = df["terrain_z_m"].to_numpy(dtype=float) - zdatum
    edges = np.linspace(np.nanmin(s), np.nanmax(s), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ib = np.digitize(s, edges) - 1
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        v = tz[ib == i]
        v = v[np.isfinite(v)]
        if len(v):
            prof[i] = np.nanmedian(v)
    good = np.isfinite(prof)
    if good.sum() >= 2:
        prof = np.interp(centers, centers[good], prof[good])
    prof = pd.Series(prof).rolling(7, center=True, min_periods=1).median().to_numpy(float)
    return centers, prof







def _panel_spatial_dood(
    ax,
    key,
    line_label,
    df,
    ref,
    h42_d_sorted,
    shared_norm,
    show_xlabel=True,
    show_ylabel=True,
    show_region_labels=True,
):
    """
    One validation-aligned vertical transect.

    Final spatial-map style:
    - show only the physically relevant lower layer:
        Askervein 0-200 m, Bolund 0-20 m;
    - keep the terrain profile;
    - use smooth interpolation, but do not extrapolate into the very high
      empty upper domain that made the earlier fills look strange.
    """
    from scipy.interpolate import griddata

    dstar = pointwise_normalized_mahalanobis(ref, df)

    s = df["s_m"].to_numpy(dtype=float)
    z = df["z-coordinate"].to_numpy(dtype=float)
    terrain = df["terrain_z_m"].to_numpy(dtype=float)
    zdatum = float(np.nanmin(terrain))
    zrel = z - zdatum

    good = np.isfinite(s) & np.isfinite(zrel) & np.isfinite(dstar)
    s = s[good]
    zrel = zrel[good]
    dstar = dstar[good]

    # Requested display cap.
    ycap = float(FINAL_SPATIAL_YMAX_BY_KEY.get(key, np.nanmax(zrel)))

    # Restrict interpolation to the displayed layer only.
    in_layer = zrel <= (ycap + 1e-9)
    s_use = s[in_layer]
    z_use = zrel[in_layer]
    d_use = dstar[in_layer]

    # Terrain profile for masking and overlay.
    ss, tt = _terrain_profile_for_selected_df(df, zdatum)

    smin, smax = float(np.nanmin(s_use)), float(np.nanmax(s_use))
    zmin, zmax = 0.0, ycap

    nx = 520
    nz = 240 if ycap > 50 else 180
    sx = np.linspace(smin, smax, nx)
    zz = np.linspace(zmin, zmax, nz)
    SX, ZZ = np.meshgrid(sx, zz)

    pts = np.column_stack([s_use, z_use])

    # Smooth field inside the displayed layer.
    G = griddata(pts, d_use, (SX, ZZ), method="linear")

    # Fill remaining holes conservatively only inside the displayed layer.
    if np.isnan(G).any():
        G_near = griddata(pts, d_use, (SX, ZZ), method="nearest")
        G = np.where(np.isnan(G), G_near, G)

    # Mask only the sub-terrain portion.
    terrain_on_grid = np.interp(sx, ss, tt)
    mask = ZZ < terrain_on_grid[None, :]
    G = np.ma.array(G, mask=mask)

    mesh = ax.pcolormesh(
        sx, zz, G,
        shading="auto",
        cmap=FINAL_CMAP,
        norm=shared_norm,
        rasterized=True,
        zorder=1,
    )

    ax.fill_between(
        ss, 0.0, tt,
        color="0.22",
        linewidth=0.0,
        zorder=8,
    )
    ax.plot(ss, tt, color="black", lw=1.15, zorder=9)

    ax.set_title(line_label, loc="left", fontsize=10.4, fontweight="bold", pad=2.0)

    if show_xlabel:
        ax.set_xlabel(r"Along-transect distance, $s$ (m)")
    else:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("Relative elevation (m)")
    else:
        ax.set_ylabel("")

    ax.set_xlim(smin, smax)
    ax.set_ylim(zmin, zmax)
    ax.grid(False)
    return mesh



def _panel_regional_dood(ax, reg, add_legend=True, title_prefix="(d) "):
    """
    Panel (d): horizontal boxplots of percentile-normalized Mahalanobis
    distance at the validation locations used by the U regional statistics.

    Horizontal orientation is intentionally used in the combined SCI figure
    because the region names are long; this avoids category-label overlap.

    Box = Q25-Q75; line = median; whiskers = Q05-Q95.
    """
    from matplotlib.patches import Patch

    if "variable" in reg.columns:
        reg = reg[reg["variable"] == "U"].copy()

    case_order = ["Askervein", "Bolund 270°", "Bolund 239°"]
    case_colors = {
        "Askervein": "#1f77b4",
        "Bolund 270°": "#ff7f0e",
        "Bolund 239°": "#9467bd",
    }
    region_orders = {
        "Askervein": ["Windward", "Crest", "Lee"],
        "Bolund 270°": ["Pre-cliff", "Post-cliff", "Plateau", "Lee"],
        "Bolund 239°": ["Pre-cliff", "Post-cliff", "Plateau", "Lee"],
    }
    label_map = {
        "Windward": "Windward",
        "Crest": "Crest",
        "Lee": "Lee-side",
        "Pre-cliff": "Pre-cliff",
        "Post-cliff": "Immediate post-cliff",
        "Plateau": "Central plateau",
    }

    stats, positions, labels, colors = [], [], [], []
    group_centers = []
    y = 1.0

    for case in case_order:
        rr = reg[reg["case"] == case].set_index("region")
        ys_case = []

        for region in region_orders[case]:
            if region not in rr.index:
                continue
            r = rr.loc[region]
            stats.append({
                "label": "",
                "whislo": float(r["DMstar_Q05"]),
                "q1": float(r["DMstar_Q25"]),
                "med": float(r["DMstar_median"]),
                "q3": float(r["DMstar_Q75"]),
                "whishi": float(r["DMstar_Q95"]),
                "fliers": [],
            })
            positions.append(y)
            labels.append(label_map[region])
            colors.append(case_colors[case])
            ys_case.append(y)
            y += 1.0

        if ys_case:
            group_centers.append((case, float(np.mean(ys_case))))
        y += 0.65

    bp = ax.bxp(
        stats,
        positions=positions,
        widths=0.62,
        vert=False,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops=dict(linewidth=0.95, edgecolor="black"),
        whiskerprops=dict(linewidth=0.95, color="0.30"),
        capprops=dict(linewidth=0.95, color="0.30"),
        medianprops=dict(linewidth=1.40, color="black"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.76)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=8.9, pad=3.0)
    ax.tick_params(axis="x", labelsize=9.4)

    ax.set_xlabel(r"Percentile-normalized $D_M^*$")
    ax.set_ylabel("")
    ax.set_title(
        title_prefix + "Validation-location OOD distributions",
        loc="left",
        fontweight="bold",
        fontsize=11.0,
    )

    # Put Askervein at the top and keep groups visually separated.
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16, lw=0.55)

    if len(reg):
        xmin = float(reg["DMstar_Q05"].min())
        xmax = float(reg["DMstar_Q95"].max())
        span = max(xmax - xmin, 0.05)
        pad = 0.08 * span
        ax.set_xlim(max(0.0, xmin - pad), min(1.0, xmax + pad))

    # Case names on the far-right inside the panel; colors match the boxes.
    xmax_plot = ax.get_xlim()[1]
    for case, yc in group_centers:
        ax.text(
            xmax_plot,
            yc,
            case,
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=case_colors[case],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
            zorder=8,
        )

    if add_legend:
        handles = [
            Patch(
                facecolor=case_colors[c],
                edgecolor="black",
                alpha=0.76,
                label=c,
            )
            for c in case_order
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="lower right",
            fontsize=8.0,
        )

def _panel_ood_vs_ml(ax, reg, gains, variable, letter, add_legends=True, include_letter_in_title=True):
    """
    Panels (e,f): controlled ML-only regional absolute RMSE change against the regional
    median normalized Mahalanobis distance evaluated at the SAME validation
    support used by that variable.

      (e) U   -> U validation locations
      (f) TKE -> TKE validation locations

    Case is encoded by color; terrain-region class by marker. No point labels.
    """
    reg_v = reg.copy()
    if "variable" in reg_v.columns:
        reg_v = reg_v[reg_v["variable"] == variable].copy()

    gain_v = gains[gains["variable"] == variable].copy()

    merged = reg_v.merge(
        gain_v,
        on=["case", "region", "variable"],
        how="inner",
    )

    case_colors = {
        "Askervein": "#1f77b4",
        "Bolund 270°": "#ff7f0e",
        "Bolund 239°": "#9467bd",
    }
    marker_map = {
        "Windward": "v",
        "Pre-cliff": "v",
        "Crest": "o",
        "Plateau": "o",
        "Post-cliff": "s",
        "Lee": "^",
    }

    for _, r in merged.iterrows():
        ax.scatter(
            r["DMstar_median"],
            r["DeltaRMSE_ML"],
            s=68,
            marker=marker_map.get(r["region"], "o"),
            facecolor=case_colors[r["case"]],
            edgecolor="black",
            linewidth=0.70,
            alpha=0.98,
            zorder=5,
        )

    ax.axhline(0.0, color="0.40", ls="--", lw=0.9)

    if len(merged):
        xmin = float(merged["DMstar_median"].min())
        xmax = float(merged["DMstar_median"].max())
        span = max(xmax - xmin, 0.04)
        pad = 0.10 * span
        ax.set_xlim(max(0.0, xmin - pad), xmax + pad)

    ax.set_xlabel(
        "Regional median\n$D_M^*$"
    )

    qlab = "U" if variable == "U" else "k"
    if variable == "U":
        ax.set_ylabel(r"$\Delta RMSE_{\mathrm{ML},U}$ (m s$^{-1}$)")
    else:
        ax.set_ylabel(r"$\Delta RMSE_{\mathrm{ML},k}$ (m$^2$ s$^{-2}$)")
    if variable == "U":
        core_title = "$\\Delta RMSE_{\\mathrm{ML},U}$\nvs OOD"
    else:
        core_title = "$\\Delta RMSE_{\\mathrm{ML},k}$\nvs OOD"
    title_text = f"({letter}) {core_title}" if include_letter_in_title else core_title
    ax.set_title(
        title_text,
        loc="left",
        fontweight="bold",
    )
    ax.grid(alpha=0.16, lw=0.55)

    from matplotlib.lines import Line2D

    case_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="none",
            markerfacecolor=case_colors[c],
            markeredgecolor="black",
            markersize=6.2,
            label=c,
        )
        for c in ["Askervein", "Bolund 270°", "Bolund 239°"]
    ]
    region_handles = [
        Line2D(
            [0], [0], marker="v", linestyle="none",
            markerfacecolor="0.75", markeredgecolor="black",
            markersize=6.2, label="Windward / pre-cliff",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none",
            markerfacecolor="0.75", markeredgecolor="black",
            markersize=6.2, label="Immediate post-cliff",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="0.75", markeredgecolor="black",
            markersize=6.2, label="Crest / central plateau",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none",
            markerfacecolor="0.75", markeredgecolor="black",
            markersize=6.2, label="Lee-side",
        ),
    ]

    if add_legends:
        leg1 = ax.legend(
            handles=case_handles,
            title="Case",
            frameon=True,
            facecolor="white",
            edgecolor="0.82",
            framealpha=0.95,
            loc="upper right",
            fontsize=7.4,
            title_fontsize=7.8,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=region_handles,
            title="Region",
            frameon=True,
            facecolor="white",
            edgecolor="0.82",
            framealpha=0.95,
            loc="lower right",
            fontsize=7.0,
            title_fontsize=7.4,
        )

def save_final_fig17_individual_panels(
    selected_field,
    ref,
    regional_dood,
    ml_gain,
    out_dir,
    dpi=FINAL_FIG_DPI,
):
    """
    Save each Fig.17 panel separately in addition to the combined 2x3 figure.

    Outputs
    -------
    individual_panels/
        Fig17a_Askervein_LineA_LineAA.png/pdf/svg
        Fig17b_Bolund270_LineA_LineB.png/pdf/svg
        Fig17c_Bolund239_LineA_LineB.png/pdf/svg
        Fig17d_Regional_Normalized_Mahalanobis.png/pdf/svg
        Fig17e_Mahalanobis_vs_ML_U.png/pdf/svg
        Fig17f_Mahalanobis_vs_ML_TKE.png/pdf/svg

    Panels (a-c) each retain BOTH validation-aligned transects and their
    reconstructed terrain/island profiles.  All three use the same
    normalized-Mahalanobis color scale.
    """
    configure_final_fig_style()
    from matplotlib.colors import Normalize

    panel_dir = out_dir / "individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    display_vmin, display_vmax = _shared_spatial_mahal_limits(selected_field, ref)
    norm = Normalize(
        vmin=display_vmin,
        vmax=display_vmax,
        clip=True,
    )
    h42_sorted = h42_mahalanobis_reference_sorted(ref)

    # --------------------------------------------------------------
    # (a-c): one atmospheric case per file, two transects per case.
    # --------------------------------------------------------------
    panel_specs = [
        (
            "a",
            "Askervein",
            FINAL_SPATIAL_GROUPS[0][1],
            "Fig17a_Askervein_LineA_LineAA",
        ),
        (
            "b",
            "Bolund 270°",
            FINAL_SPATIAL_GROUPS[1][1],
            "Fig17b_Bolund270_LineA_LineB",
        ),
        (
            "c",
            "Bolund 239°",
            FINAL_SPATIAL_GROUPS[2][1],
            "Fig17c_Bolund239_LineA_LineB",
        ),
    ]

    for letter, case_title, transects, stem in panel_specs:
        fig = plt.figure(figsize=(6.8, 7.0), facecolor="white")
        gs = fig.add_gridspec(
            2, 1,
            left=0.13, right=0.86,
            top=0.90, bottom=0.11,
            hspace=0.065,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])

        fig.suptitle(
            f"({letter}) {case_title}",
            x=0.13, y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )

        meshes = []
        for k, (ax, (key, line_label)) in enumerate(
            zip([ax_top, ax_bot], transects)
        ):
            mesh = _panel_spatial_dood(
                ax=ax,
                key=key,
                line_label=line_label,
                df=selected_field[key],
                ref=ref,
                h42_d_sorted=h42_sorted,
                shared_norm=norm,
                show_xlabel=(k == 1),
                show_ylabel=True,
                show_region_labels=True,
            )
            meshes.append(mesh)

        # Same horizontal range inside the case.
        smins = [selected_field[key]["s_m"].min() for key, _ in transects]
        smaxs = [selected_field[key]["s_m"].max() for key, _ in transects]
        xlim = (min(smins), max(smaxs))
        ax_top.set_xlim(*xlim)
        ax_bot.set_xlim(*xlim)

        # Standalone colorbar for this individual panel.
        cax = fig.add_axes([0.89, 0.18, 0.025, 0.64])
        cbar = fig.colorbar(meshes[-1], cax=cax, extend="both")
        cbar.set_label(
            r"Percentile-normalized Mahalanobis distance, $D_M^*$",
            fontsize=12.5,
            labelpad=7.0,
        )
        cbar.ax.tick_params(labelsize=11.0)
        # Spatial colorbar is zoomed to the pooled atmospheric D_M* range.

        for ext in ["png", "pdf", "svg"]:
            fp = panel_dir / f"{stem}.{ext}"
            if ext == "png":
                fig.savefig(
                    fp, dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )
            else:
                fig.savefig(
                    fp,
                    bbox_inches="tight",
                    facecolor="white",
                )
            print(f"[SAVE] {fp}")
        plt.close(fig)

    # --------------------------------------------------------------
    # (d): regional normalized Mahalanobis median + IQR.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.1, 5.2), facecolor="white")
    _panel_regional_dood(ax, regional_dood, add_legend=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.34)

    stem = "Fig17d_Regional_Normalized_Mahalanobis"
    for ext in ["png", "pdf", "svg"]:
        fp = panel_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fp, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {fp}")
    plt.close(fig)

    # --------------------------------------------------------------
    # (e): normalized Mahalanobis vs ML-only U RMSE change.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    _panel_ood_vs_ml(ax, regional_dood, ml_gain, "U", "e", add_legends=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.91, bottom=0.15)

    stem = "Fig17e_Normalized_Mahalanobis_vs_DeltaRMSE_U"
    for ext in ["png", "pdf", "svg"]:
        fp = panel_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fp, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {fp}")
    plt.close(fig)

    # --------------------------------------------------------------
    # (f): normalized Mahalanobis vs ML-only TKE RMSE change.
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    _panel_ood_vs_ml(ax, regional_dood, ml_gain, "TKE", "f", add_legends=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.91, bottom=0.15)

    stem = "Fig17f_Normalized_Mahalanobis_vs_DeltaRMSE_TKE"
    for ext in ["png", "pdf", "svg"]:
        fp = panel_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(fp, dpi=dpi, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(fp, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {fp}")
    plt.close(fig)



def make_final_fig17_2x3(
    selected_field,
    ref,
    regional_dood,
    ml_gain,
    out_dir,
    dpi=FINAL_FIG_DPI,
):
    """
    Final Sec. 6.4 main-text Fig.17 with publication-oriented spacing.

    The spatial row and statistical row use independent column widths so that
    panel (d) can be wider, while panels (a-c) remain equally sized.
    """
    configure_final_fig_style()
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    display_vmin, display_vmax = _shared_spatial_mahal_limits(selected_field, ref)
    norm = Normalize(vmin=display_vmin, vmax=display_vmax, clip=True)
    h42_sorted = h42_mahalanobis_reference_sorted(ref)

    fig = plt.figure(figsize=(15.2, 10.8), facecolor="white")

    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.50, 1.15],
        left=0.055, right=0.965,
        top=0.965, bottom=0.105,
        hspace=0.34,
    )

    top_gs = outer[0, 0].subgridspec(
        1, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.30],
        wspace=0.38,
    )
    bot_gs = outer[1, 0].subgridspec(
        1, 4,
        width_ratios=[1.24, 1.0, 1.0, 0.42],
        wspace=0.42,
    )

    meshes = []
    panel_letters = ["a", "b", "c"]

    # -------------------- (a-c): spatial fields --------------------
    for j, ((case_title, transects), letter) in enumerate(
        zip(FINAL_SPATIAL_GROUPS, panel_letters)
    ):
        inner = top_gs[0, j].subgridspec(2, 1, hspace=0.075)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0])

        display_case_title = "Askervein" if case_title == "Askervein Hill" else case_title
        ax_top.text(
            -0.08, 1.10, f"({letter}) {display_case_title}",
            transform=ax_top.transAxes,
            ha="left", va="bottom",
            fontsize=13.0, fontweight="bold",
            clip_on=False,
        )

        for k, (ax, (key, line_label)) in enumerate(zip([ax_top, ax_bot], transects)):
            mesh = _panel_spatial_dood(
                ax=ax,
                key=key,
                line_label=line_label,
                df=selected_field[key],
                ref=ref,
                h42_d_sorted=h42_sorted,
                shared_norm=norm,
                show_xlabel=(k == 1),
                show_ylabel=True,
                show_region_labels=True,
            )
            ax.tick_params(labelsize=9.6)
            ax.xaxis.label.set_size(11.0)
            ax.yaxis.label.set_size(11.0)
            meshes.append(mesh)

        smins = [selected_field[key]["s_m"].min() for key, _ in transects]
        smaxs = [selected_field[key]["s_m"].max() for key, _ in transects]
        xlim = (min(smins), max(smaxs))
        ax_top.set_xlim(*xlim)
        ax_bot.set_xlim(*xlim)

    # Shared colorbar in dedicated top-right column.
    ax_cb_host = fig.add_subplot(top_gs[0, 3])
    ax_cb_host.axis("off")
    cb_pos = ax_cb_host.get_position(fig)
    cax = fig.add_axes([
        cb_pos.x0 + 0.20 * cb_pos.width,
        cb_pos.y0 + 0.17 * cb_pos.height,
        0.27 * cb_pos.width,
        0.70 * cb_pos.height,
    ])
    cbar = fig.colorbar(meshes[-1], cax=cax, extend="both")
    cbar.set_label(
        r"Percentile-normalized $D_M^*$",
        fontsize=11.4,
        labelpad=6.0,
    )
    cbar.ax.tick_params(labelsize=10.0)

    # -------------------- (d-f): regional statistics --------------------
    axd = fig.add_subplot(bot_gs[0, 0])
    axe = fig.add_subplot(bot_gs[0, 1])
    axf = fig.add_subplot(bot_gs[0, 2])

    _panel_regional_dood(
        axd, regional_dood,
        add_legend=False,
        title_prefix="(d) ",
    )
    _panel_ood_vs_ml(
        axe, regional_dood, ml_gain, "U", "e",
        add_legends=False,
        include_letter_in_title=True,
    )
    _panel_ood_vs_ml(
        axf, regional_dood, ml_gain, "TKE", "f",
        add_legends=False,
        include_letter_in_title=True,
    )

    # Consistent readable typography in bottom row.
    for ax in [axd, axe, axf]:
        ax.tick_params(labelsize=9.6)
        ax.xaxis.label.set_size(10.8)
        ax.yaxis.label.set_size(10.9)
        ax.title.set_fontsize(11.2)
        ax.title.set_linespacing(1.15)

    # Give e/f a little breathing room around the data.
    for ax in [axe, axf]:
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        ax.set_xlim(xmin - 0.025 * span, xmax + 0.055 * span)

    # -------------------- single shared legend block --------------------
    ax_leg = fig.add_subplot(bot_gs[0, 3])
    ax_leg.axis("off")

    case_colors = {
        "Askervein": "#1f77b4",
        "Bolund 270°": "#ff7f0e",
        "Bolund 239°": "#9467bd",
    }
    case_handles = [
        Patch(facecolor=case_colors[c], edgecolor="black", alpha=0.76, label=c)
        for c in ["Askervein", "Bolund 270°", "Bolund 239°"]
    ]
    region_handles = [
        Line2D([0], [0], marker="v", linestyle="none",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=7.2, label="Windward / pre-cliff"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=7.2, label="Immediate post-cliff"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=7.2, label="Crest / central plateau"),
        Line2D([0], [0], marker="^", linestyle="none",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=7.2, label="Lee-side"),
    ]

    leg1 = ax_leg.legend(
        handles=case_handles,
        title="Case",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.96),
        frameon=False,
        fontsize=9.2,
        title_fontsize=9.7,
        borderaxespad=0.0,
        labelspacing=0.55,
        handletextpad=0.65,
    )
    ax_leg.add_artist(leg1)

    ax_leg.legend(
        handles=region_handles,
        title="Region",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.53),
        frameon=False,
        fontsize=8.8,
        title_fontsize=9.4,
        borderaxespad=0.0,
        labelspacing=0.55,
        handletextpad=0.65,
    )

    stem = "Fig17_FINAL_PERCENTILE_MAHALANOBIS_DUAL_TRANSECTS_1800dpi"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"

    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    print(f"[SAVE] {svg}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Section 6.4 ONLY: feature-space OOD analysis and figures"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder containing the nine Fluent ASCII exports",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder; default is ROOT/OOD_section6_4",
    )
    parser.add_argument("--skip-supp-figures", action="store_true")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Generate the final 2x3 Fig.17 and mast-location statistics only",
    )
    parser.add_argument(
        "--final-dpi",
        type=int,
        default=FINAL_FIG_DPI,
        help="DPI for the final PNG (default 1800)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    root = args.root
    out_root = args.out if args.out is not None else root / DEFAULT_OUTPUT_NAME

    summary_dir = out_root / "01_summary"
    fig_dir = out_root / "02_figures"
    supp_dir = out_root / "03_supplementary"

    for d in [summary_dir, fig_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SECTION 6.4 ONLY: FEATURE-SPACE OOD / TRANSFERABILITY")
    print("ROOT:", root)
    print("OUT :", out_root)
    print("=" * 100)

    paths = {k: root / name for k, name in FILE_NAMES.items()}
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"[{key}] missing: {p}")

    # -------------------------------------------------------------------------
    # 1) Read bump reference / in-family cases.
    # -------------------------------------------------------------------------
    bump: Dict[str, pd.DataFrame] = {}
    for key in ["H42", "H38", "H31"]:
        bump[key] = read_ascii(paths[key], require_z=False)
        print(f"[READ] {key:7s}: N={len(bump[key]):,}")

    ref = build_h42_reference(bump["H42"])
    print(
        "[OOD INTERPRETATION] H42 is treated as the converged calibration-case "
        "feature-space reference. The atmospheric feature exports are treated "
        "as the supplied a-priori test-state features. For a strict query-state "
        "OOD analysis of an online embedded NN, training/reference and test "
        "features should be exported at explicitly matched solver/model states."
    )
    h42_reference_table(ref).to_csv(
        summary_dir / "H42_reference_feature_statistics.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 2) Read field planes, reconstruct terrain and z_AGL.
    # -------------------------------------------------------------------------
    plane_keys = [
        "ASK_A", "ASK_AA",
        "B270_A", "B270_B",
        "B239_A", "B239_B",
    ]

    planes: Dict[str, PlaneData] = {}
    selected_field: Dict[str, pd.DataFrame] = {}

    for key in plane_keys:
        df = read_ascii(paths[key], require_z=True)
        plane = build_plane_data(key, df)
        sub = select_main_field_window(plane)

        if sub.empty:
            raise RuntimeError(
                f"{key}: main OOD selection is empty. "
                "Check coordinates/window parameters."
            )

        planes[key] = plane
        selected_field[key] = sub

        print(
            f"[SELECT] {key:7s}: "
            f"N={len(sub):,}, "
            f"s=[{sub['s_m'].min():.1f},{sub['s_m'].max():.1f}] m, "
            f"zAGL=[{sub['z_agl_m'].min():.2f},{sub['z_agl_m'].max():.2f}] m"
        )

    selected: Dict[str, pd.DataFrame] = {
        "H42": bump["H42"],
        "H38": bump["H38"],
        "H31": bump["H31"],
        **selected_field,
    }

    # -------------------------------------------------------------------------
    # FINAL Fig.17(d-f): hard-coded validation locations; no OBS file is read.
    # -------------------------------------------------------------------------
    validation_targets = build_validation_target_table()
    validation_targets.to_csv(
        summary_dir / "FINAL_validation_location_targets_HARDCODED.csv",
        index=False, encoding="utf-8-sig"
    )

    raw_validation_samples, validation_samples = (
        sample_validation_location_mahalanobis(
            selected_field=selected_field,
            ref=ref,
            targets=validation_targets,
        )
    )

    final_regional_dood = summarize_mast_location_mahalanobis(
        validation_samples
    )
    final_ml_gain = build_final_ml_gain_table()

    raw_validation_samples.to_csv(
        summary_dir / "FINAL_validation_location_D_Mstar_raw_plane_samples.csv",
        index=False, encoding="utf-8-sig"
    )
    validation_samples.to_csv(
        summary_dir / "FINAL_validation_location_percentile_D_Mstar_samples.csv",
        index=False, encoding="utf-8-sig"
    )
    validation_samples[
        [
            "case", "variable", "point", "mast", "region",
            "physical_id", "z_agl_m", "D_M_raw", "D_M_star", "D_M_Q99",
            "n_plane_representations",
        ]
    ].to_csv(
        summary_dir / "FINAL_validation_location_Mahalanobis_METHOD_AUDIT.csv",
        index=False, encoding="utf-8-sig"
    )
    final_regional_dood.to_csv(
        summary_dir / "FINAL_mast_regional_percentile_normalized_Mahalanobis.csv",
        index=False, encoding="utf-8-sig"
    )
    final_ml_gain.to_csv(
        summary_dir / "FINAL_regional_ML_only_RMSE_effect.csv",
        index=False, encoding="utf-8-sig"
    )

    # One-to-one audit table used by panels (e) and (f):
    # same case/variable/region, validation-location OOD on x and RMSE change on y.
    final_pairing_audit = final_regional_dood.merge(
        final_ml_gain, on=["case", "variable", "region"], how="inner"
    )
    final_pairing_audit.to_csv(
        summary_dir / "FINAL_Fig17ef_OOD_DeltaRMSE_pairing_AUDIT.csv",
        index=False, encoding="utf-8-sig"
    )

    # Fast path for the requested final Fig.17 only. This avoids computing
    # legacy diagnostic figures/statistics before the main 2x3 figure.
    if args.final_only:
        make_final_fig17_2x3(
            selected_field=selected_field,
            ref=ref,
            regional_dood=final_regional_dood,
            ml_gain=final_ml_gain,
            out_dir=fig_dir,
            dpi=args.final_dpi,
        )
        print("[FINAL-ONLY] Completed normalized-Mahalanobis Fig.17.")
        return

    # Additional case-level standardized mixtures for Fig.17(a-e)
    std_cases = standardized_case_feature_data(selected, ref)

    # Additional regional OOD summary for Fig.17(f-g)
    regional_df = compute_regional_ood_summary(planes, ref)
    regional_df.to_csv(
        summary_dir / "regional_OOD_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 3) Full OOD statistics for each bump case / field transect.
    # -------------------------------------------------------------------------
    rows = []
    for key in [
        "H42", "H38", "H31",
        "ASK_A", "ASK_AA",
        "B270_A", "B270_B",
        "B239_A", "B239_B",
    ]:
        rows.append(summarize_one_dataset(key, selected[key], ref))

    transect_df = pd.DataFrame(rows)

    transect_df.to_csv(
        summary_dir / "transect_OOD_COMPLETE.csv",
        index=False, encoding="utf-8-sig"
    )

    # Separate convenient tables.
    exc_cols = [
        "dataset", "N", "any_feature_exceedance_pct",
        *[f"exceed_{k}_pct" for k in FEATURE_KEYS],
    ]
    transect_df[exc_cols].to_csv(
        summary_dir / "feature_exceedance_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    w_cols = [
        "dataset", "N", "W_OOD_mean",
        *[f"W_{k}" for k in FEATURE_KEYS],
    ]
    transect_df[w_cols].to_csv(
        summary_dir / "feature_Wasserstein_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    median_cols = [
        "dataset", "N",
        *[f"median_{k}" for k in FEATURE_KEYS],
    ]
    transect_df[median_cols].to_csv(
        summary_dir / "feature_median_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    mahal_cols = [
        "dataset", "N",
        "median_Dstar", "P90_Dstar", "P95_Dstar", "P99_Dstar",
        "mean_Dstar", "max_Dstar", "fraction_Dstar_ge_0p99_pct",
        "median_DQ99", "fraction_DQ99_gt1_pct",
    ]
    transect_df[mahal_cols].to_csv(
        supp_dir / "Mahalanobis_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 4) Case-level equal-transect summary.
    # -------------------------------------------------------------------------
    case_df = make_case_level_summary(transect_df)
    case_df.to_csv(
        summary_dir / "case_level_OOD_equal_transect_weight.csv",
        index=False, encoding="utf-8-sig"
    )

    phi7_phi8_case_summary(selected).to_csv(
        summary_dir / "phi7_phi8_case_robust_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 5) Window sensitivity.
    # -------------------------------------------------------------------------
    sens_df = sensitivity_statistics(planes, ref)
    sens_df.to_csv(
        supp_dir / "height_and_window_sensitivity.csv",
        index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 6) Configuration / reproducibility.
    # -------------------------------------------------------------------------
    config = {
        "analysis_scope": "Section 6.4 feature-space applicability / OOD only",
        "input_files": FILE_NAMES,
        "features": FEATURE_COLS,
        "H42_support_quantiles": [SUPPORT_Q_LOW, SUPPORT_Q_HIGH],
        "Wasserstein_normalization": "H42 IQR feature-by-feature",
        "overall_metric": "mean of five normalized per-feature Wasserstein distances",
        "field_case_weighting": "equal transect weighting",
        "askervein_main_window": {
            "center": "automatically estimated local crest on each plane",
            "half_streamwise_m": ASK_HALF_STREAMWISE_M,
            "z_agl_max_m": ASK_ZAGL_MAX_M,
        },
        "bolund_main_window": {
            "reference": "CP",
            "s_min_m": BOL_S_MIN_M,
            "s_max_m": BOL_S_MAX_M,
            "z_agl_max_m": BOL_ZAGL_MAX_M,
        },
        "terrain_envelope": {
            "Askervein_bin_m": ASK_TERRAIN_BIN_M,
            "Bolund_bin_m": BOL_TERRAIN_BIN_M,
            "rolling_median_bins": TERRAIN_SMOOTH_WINDOW,
        },
        "reference_points_xy": REFERENCE_POINTS_XY,
        "additional_regional_bounds": {
            "Askervein": ASK_REGION_BOUNDS,
            "Bolund": BOL_REGION_BOUNDS,
        },
        "regional_ML_gain_U_source": {
            f"{k[0]} | {k[1]}": v for k, v in REGIONAL_ML_GAIN_U.items()
        },
        "Mahalanobis": {
            "role": "primary local feature-space applicability metric in Fig.17; global OOD is cross-checked by Wasserstein/support diagnostics",
            "diag_regularization": MAHALANOBIS_DIAG_REG,
            "main_normalization": "empirical H42 radial percentile: D_M^*=F_H42(D_M)=1-gamma_dm",
            "secondary_Q99_scaled_diagnostic": "D_M/Q99[D_M(H42)]",
            "Q99_quantile": MAHALANOBIS_REF_Q,
        },
    }

    with open(
        summary_dir / "analysis_configuration.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 7) Main Fig.17.
    # -------------------------------------------------------------------------
    make_final_fig17_2x3(
        selected_field=selected_field,
        ref=ref,
        regional_dood=final_regional_dood,
        ml_gain=final_ml_gain,
        out_dir=fig_dir,
        dpi=args.final_dpi,
    )

    if not args.final_only:
        make_main_figure(
            case_df=case_df,
            transect_df=transect_df,
            selected=selected,
            ref=ref,
            out_dir=fig_dir,
        )

    if not args.final_only:
        plot_standardized_feature_distributions(std_cases, fig_dir)
        plot_regional_ood_summary_and_ml_scatter(regional_df, fig_dir)

    # -------------------------------------------------------------------------
    # 8) Supplementary diagnostics.
    # -------------------------------------------------------------------------
    if (not args.skip_supp_figures) and (not args.final_only):
        plot_mahalanobis_diagnostic(transect_df, supp_dir)
        plot_feature_cdfs(selected, supp_dir)

    # -------------------------------------------------------------------------
    # 9) Text report.
    # -------------------------------------------------------------------------
    write_text_summary(
        summary_dir / "OOD_key_results.txt",
        case_df=case_df,
        transect_df=transect_df,
        ref=ref,
        planes=planes,
    )

    # -------------------------------------------------------------------------
    # 10) Console summary.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CASE-LEVEL OOD (equal-transect weighting)")
    print(
        case_df[
            ["case", "W_OOD_mean", *[f"W_{k}" for k in FEATURE_KEYS]]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nFEATURE SUPPORT EXCEEDANCE (%)")
    print(
        transect_df[
            ["dataset", *[f"exceed_{k}_pct" for k in FEATURE_KEYS]]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nFinished.")
    print("Main figure:")
    print(fig_dir / "Fig17_OOD_feature_space_transferability.png")
    print("=" * 100)


if __name__ == "__main__":
    main()
