import pandas as pd
import numpy as np

def analyze_ablation_results():
    ablation_df = pd.DataFrame({
        "Decay_Rate": [],
        "Lambda_Phys": [],
        "Seed": [],
        "lnK_RMSE": [],
        "Head_RMSE": [],
        "Mass_Balance_Error": []
    })
    optimal = ablation_df.loc[ablation_df["lnK_RMSE"].idxmin()]
    print(f"Optimal Hyperparameters: Decay Rate={optimal['Decay_Rate']}, Lambda Phys={optimal['Lambda_Phys']}")
    return ablation_df, optimal

if __name__ == "__main__":
    analyze_ablation_results()