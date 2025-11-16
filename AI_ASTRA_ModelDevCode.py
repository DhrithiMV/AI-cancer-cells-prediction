import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ------------------- Encoding Function -------------------
def encode_data(df):
    """
    Encode categorical variables and target column.
    Assumes the dataset is already cleaned.
    """
    df = df.copy()

    # Binary categorical columns
    binary_cols = ['Gender', 'Smoking', 'Alcohol_Consumption',
                   'Diabetes', 'Hypertension', 'Heart_Disease']
    encoder_bin = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')
    bin_arr = encoder_bin.fit_transform(df[binary_cols])
    bin_df = pd.DataFrame(bin_arr, columns=encoder_bin.get_feature_names_out(binary_cols), index=df.index)

    # Multi-class categorical columns
    multi_cols = ['Insurance_Type']
    encoder_multi = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
    multi_arr = encoder_multi.fit_transform(df[multi_cols])
    multi_df = pd.DataFrame(multi_arr, columns=encoder_multi.get_feature_names_out(multi_cols), index=df.index)

    # Encode target variable
    le = LabelEncoder()
    df['Target_encoded'] = le.fit_transform(df['Target_variable'])

    # Merge encoded features with rest of dataset
    df_encoded = pd.concat([df.drop(columns=binary_cols + multi_cols), bin_df, multi_df], axis=1)
    return df_encoded

# ------------------- Main Pipeline -------------------
def run_pipeline(input_csv):
    """
    Run ML pipeline:
    1) Load dataset
    2) Encode categorical columns
    3) Split data into train/test
    4) Train ML models
    5) Save predictions, metrics, plots, and pickle models
    """
    # 1) Load cleaned dataset from CSV
    data = pd.read_csv(input_csv)

    # 2) Encode categorical + target columns
    clean = encode_data(data)

    # Save processed dataset (CSV only)
    processed_csv = "processed_dataset.csv"
    clean.to_csv(processed_csv, index=False, encoding="utf-8")
    print(f"Processed dataset saved -> {processed_csv}")

    # 3) Split features and target
    X = clean.drop(columns=['Patient_ID', 'Target_variable', 'Target_encoded'], errors='ignore')
    y = clean['Target_encoded']
    patient_ids = clean['Patient_ID']

    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, patient_ids, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    metrics_records = []

    # 5) Train and evaluate models
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_records.append({
            "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "ROC_AUC": roc_auc, "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })

        # Save predictions report
        pred_df = pd.DataFrame({
            "Patient_ID": pid_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba
        })
        pred_df.to_csv(f"prediction_report_{name}.csv", index=False, encoding="utf-8")

        # Save trained model as pickle
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(clf, f)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC - {name}"); plt.legend()
        plt.savefig(f"roc_{name}.png"); plt.close()

        # Precision-Recall curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec_curve, prec_curve)
        plt.figure()
        plt.plot(rec_curve, prec_curve, label=f"AUC={pr_auc:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}"); plt.legend()
        plt.savefig(f"pr_{name}.png"); plt.close()

    # 6) Save metrics summary
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv("Model_Metrics.csv", index=False, encoding="utf-8")
    print("Metrics saved -> Model_Metrics.csv")

    # 7) Save comparison barplot
    metrics_long = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        var_name="Metric", value_name="Score"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_long)
    plt.title("Classifier Performance Comparison")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("model_comparison_barplot.png"); plt.close()
    print("Saved classifier comparison barplot -> model_comparison_barplot.png")

    return metrics_df

# ------------------- Run Script -------------------
if __name__ == "__main__":
    input_csv = "AI_Astra_CleanedDataset.csv"  # <--- your cleaned dataset file
    results = run_pipeline(input_csv)
    print(results)
