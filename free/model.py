import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statistics import mean
import warnings

#look at all cycles in the dataset, and guess for each cycle what the chemisty is and  decide  what the chemistry of the datset is
# use your trained rf model with the other datasets
# Make sure RandomForestClassifier is already trained from the previous step
from models import rf
# model.py
import sys


if len(sys.argv) < 2:
    print("Usage: python model.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


test_file_path = ['NMC_6_25C_Reg.csv','LCO_4_25C_Reg.csv','LFP_1_25C_Reg.csv']

def predict_chemistry(file_path, model):
    df = pd.read_csv(file_path)
    # Feature extraction (same as before) if you dont do manual its importante
    group = df.groupby('Cycle_Index')
    preds = []
    for cycle, g in group:
        if g['Discharge_Capacity(Ah)'].max() < 0.01:
            continue
        v = g['Voltage(V)'].values
        dcap = g['Discharge_Capacity(Ah)'].max()
        avg_v = np.mean(v)
        max_v = np.max(v)
        min_v = np.min(v)
        volt_std = np.std(v)
        energy = g['Discharge_Energy(Wh)'].max()
        feat = np.array([[avg_v, max_v, min_v, volt_std, dcap, energy]])
        
        # Apply PCA transformation if model is SVM
        if str(type(model)).find('svm') != -1:
            feat = pca.transform(feat)
        
        pred = model.predict(feat)[0]
        
        # Check if model has predict_proba method (SVM with probability=True doesn't by default)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feat)[0]
            confidence = np.max(proba)
        else:
            # For SVM without probability, use decision function as confidence proxy
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(feat)[0]
                if len(decision_scores.shape) > 0 and len(decision_scores) > 1:
                    confidence = np.max(decision_scores) / np.sum(np.abs(decision_scores))
                else:
                    confidence = abs(decision_scores) / (abs(decision_scores) + 1)  # Simple normalization
            else:
                confidence = 1.0  # Default confidence if no probability available
        
        preds.append((cycle, pred, confidence))
    
    # Show most common prediction
    if not preds:
        print(" No valid cycles found for prediction.")
        return
    
    # Majority vote
    pred_classes = [p[1] for p in preds]
    confs = [p[2] for p in preds]
    from collections import Counter
    majority = Counter(pred_classes).most_common(1)[0][0]
    avg_conf = (np.mean([conf for p, conf in zip(pred_classes, confs) if p == majority]))*100
    print(f"\n File: {os.path.basename(file_path)}")
    print(f" Predicted Chemistry: **{majority}**")
    print(f" Model Confidence: {avg_conf:.2f}") # it means how much sure it is in percentage.
    # print all cycle predictions
    print("\n Cycle-by-Cycle Predictions:")
    for cycle, pred, conf in preds:
        print(f"Cycle {cycle}: {pred} (conf: {conf:.2f})")



#usage give the unkon dataset to test:
test_file_path = ['NMC_6_25C_Reg.csv','LCO_4_25C_Reg.csv','LFP_1_25C_Reg.csv']
print("\nRandom Forest\n")

predict_chemistry(file_path, rf)

    
    





