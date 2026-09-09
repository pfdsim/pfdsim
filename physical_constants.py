"""Physical constants shared by thermodynamics, kinetics, and property models."""

# R = N_A * k_B, exact under the SI definitions (CODATA 2022).
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt
R_J_MOL_K = 8.31446261815324
R_BAR_M3_MOL_K = R_J_MOL_K / 100_000.0
R_BAR_CM3_MOL_K = R_J_MOL_K * 10.0
R_BAR_L_MOL_K = R_J_MOL_K / 100.0
R_CAL_MOL_K = R_J_MOL_K / 4.184  # Thermochemical calorie, exactly 4.184 J.
