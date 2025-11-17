import pandas as pd

# --- Load 2025 team-level data ---
df_2025 = pd.read_csv("2025.csv")

# Ensure consistent column naming
df_2025.columns = df_2025.columns.str.strip()

# Rename for formula consistency
df_2025 = df_2025.rename(columns={
    "Receiving": "Receiving_Value",
    "Oline Win Rate": "OLINE_WinRate",
    "Dline Win Rate": "DLINE_WinRate",
    "Overall Defense": "DEF_EPA_Play",
    "EPA/Rush": "Rushing_Value"
})

# --- Apply QBR and RTG regression formulas ---
df_2025["Predicted_QBR"] = (
    19.9305
    + (0.1557 * df_2025["DLINE_WinRate"])
    + (0.3097 * df_2025["OLINE_WinRate"])
    + (-2.4875 * df_2025["DEF_EPA_Play"])
    + (0.2919 * df_2025["Receiving_Value"])
    + (61.3295 * df_2025["Rushing_Value"])
)

df_2025["Predicted_RTG"] = (
    56.1008
    + (0.0650 * df_2025["DLINE_WinRate"])
    + (0.2777 * df_2025["OLINE_WinRate"])
    + (-10.2452 * df_2025["DEF_EPA_Play"])
    + (0.3663 * df_2025["Receiving_Value"])
    + (33.3639 * df_2025["Rushing_Value"])
)

# Round predictions to 3 decimals
df_2025["Predicted_QBR"] = df_2025["Predicted_QBR"].round(3)
df_2025["Predicted_RTG"] = df_2025["Predicted_RTG"].round(3)

# --- Merge with QB_25.csv ---
qb_df = pd.read_csv("QB_25.csv")
qb_df.columns = qb_df.columns.str.strip()

# Merge by Team name
merged = qb_df.merge(df_2025[["Team", "Predicted_QBR", "Predicted_RTG"]], on="Team", how="left")

# Compute over-performance columns
merged["QBR_over_Pred"] = (merged["QBR"] - merged["Predicted_QBR"]).round(3)
merged["RTG_over_Pred"] = (merged["RTG"] - merged["Predicted_RTG"]).round(3)

# --- Save final output ---
merged.to_csv("QB_25.csv", index=False)

print("✅ Predictions added and saved to QB_25.csv")
