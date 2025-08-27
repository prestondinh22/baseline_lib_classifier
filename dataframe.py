import os
import pandas as pd
import numpy as np

data_dir = "data"

all_features = [] # list of features

def extract_features(cycle_df): # function to get features

    feats = {}

    feats['Vmax'] = cycle_df["Voltage(V)"].max() #VMAX 
    feats['Vmin'] = cycle_df["Voltage(V)"].min() #VMIN
    feats['Vrange'] = feats['Vmax'] - feats['Vmin'] #VRANGE

    charge_mask = cycle_df["Current(A)"] > 0 
    discharge_mask = cycle_df["Current(A)"] < 0
    rest_mask = cycle_df["Current(A)"] == 0
    
  
    if discharge_mask.any():
        rest_after_discharge = cycle_df.loc[
            rest_mask & (cycle_df["Test_Time(s)"] > cycle_df[discharge_mask]["Test_Time(s)"].max()), 
            "Voltage(V)"
        ].mean()
    else:
        rest_after_discharge = np.nan
    feats['rest_voltage_after_discharge'] = rest_after_discharge
    feats['dvdt_discharge'] = cycle_df.loc[discharge_mask, "dV/dt(V/s)"].mean()

    

    return feats

for chem in os.listdir(data_dir): #looping through chemistry
    chem_path = os.path.join(data_dir, chem)
    for cell_file in os.listdir(chem_path): #looping through cell
        cell_path = os.path.join(chem_path, cell_file)
        df = pd.read_csv(cell_path)
        for cycle_id, cycle_df in df.groupby("Cycle_Index"):
            feats = extract_features(cycle_df)
            feats["chemistry"] = chem
            
            all_features.append(feats)

features_df = pd.DataFrame(all_features)
print(features_df.head)