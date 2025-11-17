import pandas as pd
import statsmodels.api as sm
import os

# --- Step 1: Load all files from Training Data directory ---
base_dir = "Training Data"
files = {
    "DLINE": os.path.join(base_dir, "DLINE (Overall Win Rate).csv"),
    "OLINE": os.path.join(base_dir, "OlINE (Overall Win Rate).csv"),
    "DEF_EPA": os.path.join(base_dir, "Overall Defense (EPA_Play).csv"),
    "RECV": os.path.join(base_dir, "Receiving.csv"),   # raw values (not rank)
    "RUSH": os.path.join(base_dir, "Rushing.csv"),     # raw values (not rank)
    "QB": os.path.join(base_dir, "QB.csv")
}

# --- Step 2: Helper function to load and clean team-level files ---
def load_team_file(path, var_name):
    df = pd.read_csv(path)
    team_col = [c for c in df.columns if c.strip().lower() == "team"][0]
    df = df.melt(id_vars=[team_col], var_name="Year", value_name=var_name)
    df.rename(columns={team_col: "Team"}, inplace=True)

    df[var_name] = df[var_name].astype(str).str.replace("%", "", regex=False)
    df[var_name] = df[var_name].str.replace(",", "", regex=False)
    df[var_name] = pd.to_numeric(df[var_name], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    return df

# --- Step 3: Load all team-level data ---
dline = load_team_file(files["DLINE"], "DLINE_WinRate")
oline = load_team_file(files["OLINE"], "OLINE_WinRate")
def_epa = load_team_file(files["DEF_EPA"], "DEF_EPA_Play")
recv = load_team_file(files["RECV"], "Receiving_Value")
rush = load_team_file(files["RUSH"], "Rushing_Value")

# --- Step 4: Load QB data ---
qb = pd.read_csv(files["QB"])
qb["QBR"] = pd.to_numeric(qb["QBR"], errors="coerce")
qb["RTG"] = pd.to_numeric(qb["RTG"], errors="coerce")
qb["Year"] = pd.to_numeric(qb["Year"], errors="coerce")

# --- Step 5: Merge all datasets ---
merged = qb.merge(dline, on=["Team", "Year"], how="left") \
           .merge(oline, on=["Team", "Year"], how="left") \
           .merge(def_epa, on=["Team", "Year"], how="left") \
           .merge(recv, on=["Team", "Year"], how="left") \
           .merge(rush, on=["Team", "Year"], how="left")

merged = merged.dropna(subset=["DLINE_WinRate", "OLINE_WinRate", "DEF_EPA_Play",
                               "Receiving_Value", "Rushing_Value"])

# --- Step 6: Prepare predictors ---
predictors = ["DLINE_WinRate", "OLINE_WinRate", "DEF_EPA_Play", "Receiving_Value", "Rushing_Value"]

# --- Step 7: Train regression models ---
def train_regression(y_var):
    X = merged[predictors]
    y = merged[y_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

model_qbr = train_regression("QBR")
model_rtg = train_regression("RTG")

# --- Step 8: Predict QBR and RTG ---
merged["Predicted_QBR"] = model_qbr.predict(sm.add_constant(merged[predictors]))
merged["Predicted_RTG"] = model_rtg.predict(sm.add_constant(merged[predictors]))

# --- Step 9: Calculate over/under-performance ---
merged["QBR_over_Pred"] = merged["QBR"] - merged["Predicted_QBR"]
merged["RTG_over_Pred"] = merged["RTG"] - merged["Predicted_RTG"]

# --- Step 10: Round numeric columns ---
numeric_cols = ["QBR", "Predicted_QBR", "QBR_over_Pred",
                "RTG", "Predicted_RTG", "RTG_over_Pred"]
merged[numeric_cols] = merged[numeric_cols].round(2)

# --- Step 11: Save main QB data ---
output_cols = ["Team", "Year", "Name", "QBR", "Predicted_QBR", "QBR_over_Pred",
               "RTG", "Predicted_RTG", "RTG_over_Pred"]
output = merged[output_cols]
output.to_csv("QB Data.csv", index=False)

# --- Step 12: Identify top and bottom performers ---
top_qbr = merged.sort_values("QBR_over_Pred", ascending=False).head(10)
bottom_qbr = merged.sort_values("QBR_over_Pred").head(10)

print("\n🔥 Top Overperformers (QBR):")
print(top_qbr[["Team", "Year", "QBR", "Predicted_QBR", "QBR_over_Pred"]].round(2))

print("\n💀 Biggest Underperformers (QBR):")
print(bottom_qbr[["Team", "Year", "QBR", "Predicted_QBR", "QBR_over_Pred"]].round(2))

print("✅ Predictions complete. Results saved to QB Data.csv")

# --- Step 13: Compute standardized betas (impact strength) ---
def standardized_betas(model, X, y):
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    model_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    return model_std.params

X_qbr = merged[predictors]
y_qbr = merged["QBR"]
X_rtg = merged[predictors]
y_rtg = merged["RTG"]

std_betas_qbr = standardized_betas(model_qbr, X_qbr, y_qbr)
std_betas_rtg = standardized_betas(model_rtg, X_rtg, y_rtg)

# --- Step 14: Create and save Analysis.csv ---
analysis = pd.DataFrame({
    "Variable": predictors,
    "Impact_on_QBR": model_qbr.params[predictors].round(4),
    "P_value_QBR": model_qbr.pvalues[predictors].round(4),
    "Std_Beta_QBR": std_betas_qbr[predictors].round(4),
    "Impact_on_RTG": model_rtg.params[predictors].round(4),
    "P_value_RTG": model_rtg.pvalues[predictors].round(4),
    "Std_Beta_RTG": std_betas_rtg[predictors].round(4)
})
analysis.to_csv("Analysis.csv", index=False)

# --- Step 15: Save model formulas (with intercept + R²) ---
formula_qbr = f"QBR = {model_qbr.params['const']:.4f}"
for var in predictors:
    coef = model_qbr.params[var]
    formula_qbr += f" + ({coef:.4f} * {var})"

formula_rtg = f"RTG = {model_rtg.params['const']:.4f}"
for var in predictors:
    coef = model_rtg.params[var]
    formula_rtg += f" + ({coef:.4f} * {var})"

with open("Model_Formulas.txt", "w") as f:
    f.write("--- QBR Model Formula ---\n")
    f.write(f"{formula_qbr}\n")
    f.write(f"R-squared: {model_qbr.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {model_qbr.rsquared_adj:.4f}\n\n")

    f.write("--- RTG Model Formula ---\n")
    f.write(f"{formula_rtg}\n")
    f.write(f"R-squared: {model_rtg.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {model_rtg.rsquared_adj:.4f}\n")

print("\n📊 Variable impact analysis saved to 'Analysis.csv'")
print("🧮 Model formulas (with intercepts + R²) saved to 'Model_Formulas.txt'")
