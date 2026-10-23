"""
Generate Realistic House Price Dataset
Generates synthetic yet realistic real estate data for Indian metro locations.
Features:
- Location
- Total Area (sq. ft.)
- Bedrooms (BHK)
- Bathrooms
- Property Age (Years)
- Parking (0 or 1)
- Price (in Lakhs INR)
"""

import numpy as np
import pandas as pd
import os

def generate_house_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    
    locations = [
        "Whitefield", "Indiranagar", "Koramangala", "HSR Layout", 
        "Electronic City", "Marathahalli", "Bellandur", "Jayanagar", 
        "Hebbal", "Malleshwaram"
    ]
    
    # Base price per sqft for each location (in INR)
    location_rates = {
        "Electronic City": 4500,
        "Marathahalli": 6200,
        "Whitefield": 6800,
        "Bellandur": 7500,
        "Hebbal": 8200,
        "HSR Layout": 9500,
        "Malleshwaram": 11000,
        "Jayanagar": 12500,
        "Indiranagar": 15000,
        "Koramangala": 16500
    }
    
    data = []
    
    for _ in range(n_samples):
        loc = np.random.choice(locations)
        rate_per_sqft = location_rates[loc]
        
        # BHK distribution: 1 to 5
        bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.45, 0.30, 0.12, 0.03])
        
        # Realistic sqft depending on BHK
        if bhk == 1:
            sqft = np.random.randint(450, 750)
            bathrooms = 1
        elif bhk == 2:
            sqft = np.random.randint(800, 1300)
            bathrooms = np.random.choice([1, 2], p=[0.2, 0.8])
        elif bhk == 3:
            sqft = np.random.randint(1300, 2100)
            bathrooms = np.random.choice([2, 3], p=[0.3, 0.7])
        elif bhk == 4:
            sqft = np.random.randint(2100, 3200)
            bathrooms = np.random.choice([3, 4], p=[0.3, 0.7])
        else: # 5 BHK
            sqft = np.random.randint(3000, 4800)
            bathrooms = np.random.choice([4, 5], p=[0.4, 0.6])
            
        age = np.random.randint(0, 22)
        parking = np.random.choice([0, 1], p=[0.25, 0.75])
        
        # Calculate price in Lakhs (1 Lakh = 100,000 INR)
        # Base price from sqft and location rate
        base_price_inr = sqft * rate_per_sqft
        
        # Additional amenities & factors
        bathroom_bonus = (bathrooms - 1) * 75000
        parking_bonus = parking * 200000
        
        # Age depreciation: roughly 0.8% per year up to 20%
        depreciation = min(age * 0.008, 0.20)
        
        # Non-linear premium for luxury large properties (>2500 sqft)
        luxury_factor = 1.08 if sqft > 2500 else 1.0
        
        total_price_inr = (base_price_inr + bathroom_bonus + parking_bonus) * (1 - depreciation) * luxury_factor
        
        # Adding realistic Gaussian noise (+/- 5%)
        noise = np.random.normal(0, 0.04 * total_price_inr)
        final_price_inr = total_price_inr + noise
        
        # Convert to Lakhs and round to 2 decimals
        price_lakhs = round(max(15.0, final_price_inr / 100000.0), 2)
        
        data.append({
            "location": loc,
            "total_sqft": sqft,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "property_age": age,
            "parking": parking,
            "price_lakhs": price_lakhs
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    os.makedirs("d:/3sem/ML projects/house_price_prediction/dataset", exist_ok=True)
    df = generate_house_data(n_samples=2500)
    output_path = "d:/3sem/ML projects/house_price_prediction/dataset/house_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records successfully saved to {output_path}")
    print("\nDataset Preview:")
    print(df.head(5))
    print("\nDataset Summary:")
    print(df.describe())
