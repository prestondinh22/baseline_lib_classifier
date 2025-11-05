#!/usr/bin/env python
# coding: utf-8

# In[60]:





# In[2]:


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
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

path_LFP = Path('clean_data/LFP_5_15_35_45_degC')
path_LCO = Path("clean_data/LCO_5_15_35_45_degC")
path_NMC = Path("clean_data/NMC_5_15_35_45_degC")

if __name__ == "__main__":

    def read_csv_until_value(file_path): 
        all_data = []
        for subfolder in file_path.iterdir():
            if subfolder.is_dir():
                for chemistry_file in subfolder.iterdir():
                    if chemistry_file.suffix == '.csv': #find each csv
                        df = pd.read_csv(chemistry_file)
                        mask = df['Current(A)'] != 0
                        final = df[mask].copy()
                        if not final.empty:
                            final['Identifier'] = chemistry_file.stem[4:8].upper() #unique identifier
                            final['Chemistry'] = chemistry_file.stem[:3].upper()
                            final = final[~final['Identifier'].str.contains('_45|_5C')]
                            all_data.append(final)
        return pd.concat(all_data, ignore_index=True)
    #combine all
    test = pd.concat([read_csv_until_value(path_LFP), read_csv_until_value(path_NMC), read_csv_until_value(path_LCO)], ignore_index=True)
    print(test.shape)

    # Get unique chemistries
    chemistries = test['Chemistry'].unique()

    # Create 3 separate plots
    for chem in chemistries:
        plt.figure(figsize=(10, 6))

        # Get data for this chemistry
        chem_data = test[test['Chemistry'] == chem]

        # Get first 5 identifiers
        identifiers = chem_data['Identifier'].unique()[:3]  # Changed back to 5 for readability

        # Plot each cycle
        for identifier in identifiers:
            cycle_data = chem_data[chem_data['Identifier'] == identifier]  # ✅ ADD THIS LINE
            plt.plot(cycle_data['Test_Time(s)'], cycle_data['Voltage(V)'], label=identifier, alpha=0.7)

        plt.xlabel('Test Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'{chem} Chemistry - Voltage vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


    # In[63]:


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
    from sklearn.tree import DecisionTreeClassifier

    path = Path('data')
    def combine_csv_features(other, path):
        all_dfs = []

        for subfolder in path.iterdir():
            if subfolder.is_dir():
                for chemistry_file in subfolder.iterdir():
                    if chemistry_file.suffix == '.csv':

                        df = pd.read_csv(chemistry_file)

                        # Apply negative current filter
                        mask = df['Current(A)'] != 0
                        df = df[mask].copy()

                        if not df.empty:  # Only add if we have filtered data
                            chemistry_name = chemistry_file.stem[:3].upper()
                            identifier = chemistry_file.stem[4].upper()
                            df['Identifier'] = identifier
                            df['Chemistry'] = chemistry_name
                            df = df[~df['Identifier'].str.contains('_45|_5C')]
                            all_dfs.append(df)

        if all_dfs:
            return pd.concat([other, pd.concat(all_dfs, ignore_index=True)], ignore_index=True)
        else:
            return other  # Return original if no new data found


    df = combine_csv_features(test, path)  # One big DataFrame with Chemistry column
    print(df.shape)

    #paths: you can put all the files in the foder and orgize together to define the column chemistry (to have data labeled)

    def extract_features(df, plot=True):
        cycle_groups = df.groupby(['Chemistry', 'Cycle_Index', 'Identifier'])

        features = []

        # For plotting: collect data by chemistry
        plot_data = {}

        for (chem, cycle, identifier), group in cycle_groups: 

            v = group['Voltage(V)'].values
            c = group['Discharge_Capacity(Ah)'].values
            t = group['Test_Time(s)'].values
            dcap = group['Discharge_Capacity(Ah)'].max()
            avg_v = np.mean(v)
            max_v = np.max(v)
            min_v = np.min(v)
            volt_std = np.std(v)
            energy = group['Discharge_Energy(Wh)'].max()

            features.append({
                'chemistry': chem,
                'cycle': cycle,
                'avg_voltage': avg_v,
                'max_voltage': max_v,
                'min_voltage': min_v,
                'volt_std': volt_std,
                'discharge_capacity': dcap,
                'discharge_energy': energy,
            })

            # Collect data for plotting
            if plot:
                if chem not in plot_data:
                    plot_data[chem] = []
                # Only keep first 5 cycles per chemistry for cleaner plots
                if len(plot_data[chem]) < 50:
                    norm_t = (t - np.min(t)) / (np.max(t) - np.min(t)) if np.max(t) > np.min(t) else np.zeros_like(t)
                    plot_data[chem].append((v, norm_t, cycle, identifier))

        # Create plots
        if plot:
            for chem in plot_data.keys():
                plt.figure(figsize=(10, 6))

                for v, t, cycle, identifier in plot_data[chem]:
                    plt.plot(t, v, alpha=0.7)

                plt.xlabel('Test_Time(s)')
                plt.ylabel('Voltage (V)')
                plt.title(f'{chem} Chemistry - Voltage vs Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

        return pd.DataFrame(features)


    # Usage
    features_df = extract_features(df, plot=True)
    print(features_df)


    # Get unique chemistries

    X = features_df.drop(columns=['chemistry', 'cycle'])
    y = features_df['chemistry']
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)

    # Decision tree

    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )


    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_val)



    # === 2. PCA + SVM ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf')
    svm.fit(X_train_pca, y_train_pca)
    svm_preds = svm.predict(X_val_pca)

    # 3. Logreg

    logreg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegressionCV(
            scoring='accuracy',
            cv=5,
            max_iter=1000
        ))
    ])

    logreg_pipe.fit(X_train, y_train)
    logreg_preds = logreg_pipe.predict(X_val)


    # === EVALUATION ===
    def plot_conf_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.show()

    print("\nRandom Forest Report")
    print(classification_report(y_val, rf_preds))
    plot_conf_matrix(y_val, rf_preds, "Random Forest")
    print("Best params (Random Forest):", rf.get_params())
    # Get feature importances from your trained RF model
    feature_importances = rf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    print(importance_df)


    print("\nPCA + SVM Report")
    print(classification_report(y_val_pca, svm_preds))
    plot_conf_matrix(y_val_pca, svm_preds, "PCA + SVM")

    print("\nLogistic Regression Report")
    print(classification_report(y_val, logreg_preds))
    plot_conf_matrix(y_val, logreg_preds, "Logistic Regression") 

    print("\nDecision Tree Report")
    print(classification_report(y_val, dt_preds))
    plot_conf_matrix(y_val, dt_preds, "Decision Tree") 








    # In[ ]:





    # In[65]:


    import pandas as pd
    import numpy as np
    import os
    #look at all cycles in the dataset, and guess for each cycle what the chemisty is and  decide  what the chemistry of the datset is
    # use your trained rf model with the other datasets
    # Make sure RandomForestClassifier is already trained from the previous step

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
    for path in test_file_path:
        predict_chemistry(path, rf)

    print("\nLogreg\n")
    for path in test_file_path:
        predict_chemistry(path, logreg_pipe)

    print("\nDecision Tree\n")
    for path in test_file_path:
        predict_chemistry(path, dt)

    print("\nSVM+PCA\n")
    for path in test_file_path:
        predict_chemistry(path, svm)



import joblib

# Save trained models
joblib.dump(rf, "rf_trained.joblib")




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




