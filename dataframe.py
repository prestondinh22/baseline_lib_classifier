import os
import pandas as pd
import numpy as np

data_dir = "data"

all_features = []

def extract_features(cycle_df):

    feats = {}

    feats['Vmax'] = cycle_df["Voltage(V)"].max()
    feats['Vmin'] = cycle_df["Voltage(V)"].min()
    feats['Vrange'] = feats['Vmax'] - feats['Vmin']

    charge_mask = cycle_df["Current(A)"] > 0
    discharge_mask = cycle_df["Current(A)"] < 0
    rest_mask = cycle_df["Current(A)"] == 0
    cycle_df['temperature'] = cycle_df[["Temperature (C)_1","Temperature (C)_2","Temperature (C)_3","Temperature (C)_4"]].mean(axis=1)

    feats['avg_temp_charge'] = cycle_df.loc[charge_mask, "temperature"].mean()
    feats['avg_temp_discharge'] = cycle_df.loc[discharge_mask, "temperature" ].mean()
    feats['avg_temp_rest'] = cycle_df.loc[rest_mask, "temperature"].mean()
  
    if discharge_mask.any():
        rest_after_discharge = cycle_df.loc[
            rest_mask & (cycle_df["Test_Time(s)"] > cycle_df[discharge_mask]["Test_Time(s)"].max()), 
            "Voltage(V)"
        ].mean()
    else:
        rest_after_discharge = np.nan
    feats['rest_voltage_after_discharge'] = rest_after_discharge
    feats['dvdt_discharge'] = cycle_df.loc[discharge_mask, "dV/dt(V/s)"].mean()

    discharge = cycle_df.loc[discharge_mask] #array of values during discharge

    if not discharge.empty:
        discharge_capacity = discharge["Discharge_Capacity(Ah)"].values #x axis
        voltage = discharge["Voltage(V)"].values #y axis
        soc = (discharge_capacity - discharge_capacity.min()) / (discharge_capacity.max() - discharge_capacity.min()) #normalize soc
        target_soc = np.linspace(0.1,0.9,5) #evenly distributed SOC points

        sampled_voltages = np.interp(target_soc, soc, voltage)
    else:
        sampled_voltages = [np.nan]*5

    for i, v in enumerate(sampled_voltages, 1):
        feats[f"V_SOC_{i}"] = v

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