"""
Train Machine Learning Models for House Price Prediction
Algorithms:
1. Linear Regression
2. Random Forest Regressor

Features:
- location (Categorical)
- total_sqft (Numerical)
- bhk (Numerical)
- bathrooms (Numerical)
- property_age (Numerical)
- parking (Numerical binary)

Saves trained pipelines, performance metrics, and visualization plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for saving charts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "house_data.csv")
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "static", "images")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Feature & Target Separation
    X = df[["location", "total_sqft", "bhk", "bathrooms", "property_age", "parking"]]
    y = df["price_lakhs"]
    
    categorical_features = ["location"]
    numerical_features = ["total_sqft", "bhk", "bathrooms", "property_age", "parking"]
    
    locations_list = sorted(df["location"].unique().tolist())
    print(f"[INFO] Found {len(locations_list)} unique locations: {locations_list}")
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Preprocessing Pipeline
    # Using handle_unknown='ignore' so unseen locations don't throw errors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    
    # Tree preprocessor doesn't strictly need standard scaling, but OneHotEncoder is needed
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )
    
    # 1. Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # 2. Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    print("\n[TRAINING] Fitting Linear Regression model...")
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    
    print("[TRAINING] Fitting Random Forest Regressor model...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    
    # Evaluation Metrics calculation
    def compute_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "mape": round(float(mape), 2)
        }
    
    lr_metrics = compute_metrics(y_test, lr_preds)
    rf_metrics = compute_metrics(y_test, rf_preds)
    
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE COMPARISON           ")
    print("="*50)
    print(f"Metric\t\tLinear Regression\tRandom Forest")
    print(f"R2 Score:\t{lr_metrics['r2_score']:.4f}\t\t\t{rf_metrics['r2_score']:.4f}")
    print(f"MAE (Rs. Lakhs):\t{lr_metrics['mae']:.2f}\t\t\t{rf_metrics['mae']:.2f}")
    print(f"RMSE (Rs. Lakhs):\t{lr_metrics['rmse']:.2f}\t\t\t{rf_metrics['rmse']:.2f}")
    print(f"MAPE (% Error):\t{lr_metrics['mape']:.2f}%\t\t\t{rf_metrics['mape']:.2f}%")
    print("="*50)
    
    # Save Models
    lr_path = os.path.join(models_dir, "linear_regression_pipeline.pkl")
    rf_path = os.path.join(models_dir, "random_forest_pipeline.pkl")
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_pipeline, rf_path)
    print(f"\n[SAVED] Linear Regression model saved to: {lr_path}")
    print(f"[SAVED] Random Forest model saved to: {rf_path}")
    
    # Extract Feature Importances from Random Forest
    rf_model = rf_pipeline.named_steps["regressor"]
    rf_prep = rf_pipeline.named_steps["preprocessor"]
    cat_encoder = rf_prep.named_transformers_["cat"]
    cat_names = [f"loc_{cat}" for cat in cat_encoder.categories_[0]]
    all_feature_names = numerical_features + cat_names
    importances = rf_model.feature_importances_
    
    feature_importance_data = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    # Save Metadata and Metrics
    metadata = {
        "locations": locations_list,
        "features": {
            "numerical": numerical_features,
            "categorical": categorical_features
        },
        "metrics": {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics
        },
        "feature_importances": feature_importance_data[:10],
        "dataset_stats": {
            "total_records": len(df),
            "min_sqft": int(df["total_sqft"].min()),
            "max_sqft": int(df["total_sqft"].max()),
            "min_price_lakhs": float(df["price_lakhs"].min()),
            "max_price_lakhs": float(df["price_lakhs"].max()),
            "avg_price_lakhs": round(float(df["price_lakhs"].mean()), 2)
        }
    }
    
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Metadata and metrics saved to: {os.path.join(models_dir, 'metadata.json')}")
    
    # ----------------- Visualizations ----------------- #
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Actual vs Predicted Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear Regression Plot
    axes[0].scatter(y_test, lr_preds, color="#3b82f6", alpha=0.55, edgecolors="none", s=35)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Line (y=x)")
    axes[0].set_title(f"Linear Regression\n(R2 = {lr_metrics['r2_score']:.3f}, MAE = Rs. {lr_metrics['mae']}L)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Actual Price (Rs. Lakhs)", fontsize=11)
    axes[0].set_ylabel("Predicted Price (Rs. Lakhs)", fontsize=11)
    axes[0].legend()
    
    # Random Forest Plot
    axes[1].scatter(y_test, rf_preds, color="#10b981", alpha=0.55, edgecolors="none", s=35)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Line (y=x)")
    axes[1].set_title(f"Random Forest Regressor\n(R2 = {rf_metrics['r2_score']:.3f}, MAE = Rs. {rf_metrics['mae']}L)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Actual Price (Rs. Lakhs)", fontsize=11)
    axes[1].set_ylabel("Predicted Price (Rs. Lakhs)", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    pred_plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plt.savefig(pred_plot_path, dpi=200)
    plt.close()
    print(f"[SAVED] Plot saved: {pred_plot_path}")
    
    # 2. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_data[:8]
    feat_names = [f["feature"] for f in top_features][::-1]
    feat_scores = [f["importance"] * 100 for f in top_features][::-1]
    
    colors = sns.color_palette("viridis", len(feat_names))
    plt.barh(feat_names, feat_scores, color=colors)
    plt.title("Top Feature Importances (Random Forest)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Importance Percentage (%)", fontsize=11)
    for i, v in enumerate(feat_scores):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontweight='semibold')
    plt.xlim(0, max(feat_scores) + 8)
    plt.tight_layout()
    feat_plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()
    print(f"[SAVED] Plot saved: {feat_plot_path}")
    
    # 3. Model Comparison Metrics Chart
    plt.figure(figsize=(8, 5))
    metrics_names = ["R2 Score (x100)", "MAE (Rs. Lakhs)", "RMSE (Rs. Lakhs)"]
    lr_vals = [lr_metrics["r2_score"] * 100, lr_metrics["mae"], lr_metrics["rmse"]]
    rf_vals = [rf_metrics["r2_score"] * 100, rf_metrics["mae"], rf_metrics["rmse"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, lr_vals, width, label='Linear Regression', color='#3b82f6')
    plt.bar(x + width/2, rf_vals, width, label='Random Forest', color='#10b981')
    
    plt.title("Model Performance Metrics Comparison", fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_names, fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend()
    plt.tight_layout()
    comp_plot_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"[SAVED] Plot saved: {comp_plot_path}")
    
    print("\n[SUCCESS] Model training and evaluation complete!")

if __name__ == "__main__":
    train_and_evaluate()
