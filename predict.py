"""
House Price Prediction - Inference Module & CLI
Supports single and batch prediction using trained models:
1. Linear Regression
2. Random Forest Regressor
"""

import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models and metadata
LR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression_pipeline.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_pipeline.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

def load_resources():
    if not os.path.exists(LR_MODEL_PATH) or not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError("Trained models not found! Run train_model.py first.")
    
    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    return lr_model, rf_model, metadata

def predict_house_price(location: str, total_sqft: float, bhk: int, bathrooms: int, 
                        property_age: int = 5, parking: int = 1):
    """
    Predicts house price using both Linear Regression and Random Forest Regressor.
    Returns dictionary with predicted prices (in Lakhs and Crores/Rupees) and model metrics.
    """
    lr_model, rf_model, metadata = load_resources()
    
    # Create input dataframe matching model expected schema
    input_df = pd.DataFrame([{
        "location": location,
        "total_sqft": float(total_sqft),
        "bhk": int(bhk),
        "bathrooms": int(bathrooms),
        "property_age": int(property_age),
        "parking": int(parking)
    }])
    
    lr_pred_lakhs = float(lr_model.predict(input_df)[0])
    rf_pred_lakhs = float(rf_model.predict(input_df)[0])
    
    # Clip negative or unrealistic low predictions if any
    lr_pred_lakhs = max(10.0, round(lr_pred_lakhs, 2))
    rf_pred_lakhs = max(10.0, round(rf_pred_lakhs, 2))
    
    def format_price(lakhs):
        inr = lakhs * 100000
        if lakhs >= 100:
            crores = lakhs / 100
            return f"Rs. {crores:.2f} Cr (Rs. {lakhs:.2f} Lakhs)"
        return f"Rs. {lakhs:.2f} Lakhs"
        
    avg_price_lakhs = round((lr_pred_lakhs + rf_pred_lakhs) / 2, 2)
    price_per_sqft = round((rf_pred_lakhs * 100000) / total_sqft, 2)
    
    return {
        "inputs": {
            "location": location,
            "total_sqft": total_sqft,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "property_age": property_age,
            "parking": "Yes" if parking == 1 else "No"
        },
        "predictions": {
            "random_forest": {
                "price_lakhs": rf_pred_lakhs,
                "formatted_price": format_price(rf_pred_lakhs),
                "r2_score": metadata["metrics"]["random_forest"]["r2_score"],
                "mae": metadata["metrics"]["random_forest"]["mae"]
            },
            "linear_regression": {
                "price_lakhs": lr_pred_lakhs,
                "formatted_price": format_price(lr_pred_lakhs),
                "r2_score": metadata["metrics"]["linear_regression"]["r2_score"],
                "mae": metadata["metrics"]["linear_regression"]["mae"]
            },
            "ensemble_average": {
                "price_lakhs": avg_price_lakhs,
                "formatted_price": format_price(avg_price_lakhs)
            },
            "price_per_sqft": price_per_sqft
        }
    }

def interactive_cli():
    _, _, metadata = load_resources()
    locations = metadata["locations"]
    
    print("\n" + "="*55)
    print("        HOUSE PRICE PREDICTION - CLI DEMO       ")
    print("="*55)
    print("Available Locations:")
    for idx, loc in enumerate(locations, 1):
        print(f" {idx}. {loc}", end="\t" if idx % 2 != 0 else "\n")
    print("\n" + "-"*55)
    
    try:
        loc_choice = int(input(f"Select Location number (1-{len(locations)}): "))
        selected_location = locations[loc_choice - 1]
        
        sqft = float(input("Enter Total Area in sq.ft (e.g. 1200): "))
        bhk = int(input("Enter Bedrooms (BHK, e.g. 2): "))
        bathrooms = int(input("Enter Bathrooms (e.g. 2): "))
        age = int(input("Enter Property Age in years (e.g. 3): "))
        parking = int(input("Parking available? (1 for Yes, 0 for No): "))
        
        result = predict_house_price(selected_location, sqft, bhk, bathrooms, age, parking)
        
        print("\n" + "="*55)
        print("                 ESTIMATION RESULTS              ")
        print("="*55)
        print(f"Property: {bhk} BHK in {selected_location} ({sqft} sq.ft, {age} yrs old)")
        print(f"\n[1] Random Forest Prediction (Best Model):")
        print(f"    Estimated Price: {result['predictions']['random_forest']['formatted_price']}")
        print(f"    Model Accuracy (R2 Score): {result['predictions']['random_forest']['r2_score'] * 100:.1f}%")
        
        print(f"\n[2] Linear Regression Prediction:")
        print(f"    Estimated Price: {result['predictions']['linear_regression']['formatted_price']}")
        print(f"    Model Accuracy (R2 Score): {result['predictions']['linear_regression']['r2_score'] * 100:.1f}%")
        
        print(f"\nEstimated Rate per sq.ft: Rs. {result['predictions']['price_per_sqft']}/sq.ft")
        print("="*55)
        
    except (ValueError, IndexError) as e:
        print(f"\n[ERROR] Invalid input provided: {e}")

if __name__ == "__main__":
    # Test sample prediction automatically
    sample = predict_house_price(
        location="Indiranagar",
        total_sqft=1450,
        bhk=3,
        bathrooms=2,
        property_age=4,
        parking=1
    )
    print("Sample Prediction Check (Indiranagar 3 BHK 1450 sq.ft):")
    print(f"Random Forest Price: {sample['predictions']['random_forest']['formatted_price']}")
    print(f"Linear Regression Price: {sample['predictions']['linear_regression']['formatted_price']}")
    print(f"Price per sq.ft: Rs. {sample['predictions']['price_per_sqft']}/sq.ft")
