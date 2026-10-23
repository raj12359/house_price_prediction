# 🏡 EstateIQ: House Price Prediction System (Machine Learning)

**Difficulty:** Easy | **Semester:** 3rd Sem ML Project  
**Algorithms:** Linear Regression & Random Forest Regressor  
**Features:** Location, Super Built-up Area (sq. ft.), Bedrooms (BHK), Bathrooms, Property Age, Reserved Parking.

---

## 📌 Project Overview
EstateIQ is an end-to-end Machine Learning project designed to accurately estimate residential house and apartment prices. It provides side-by-side predictions comparing:
1. **Linear Regression** (Parametric Baseline Model)
2. **Random Forest Regressor** (Non-linear Ensemble Bagging Model)

The project includes:
- Realistic synthetic dataset generation with 2,500 real estate listings across prime metro zones.
- Data preprocessing with Scikit-Learn `ColumnTransformer` (One-Hot Encoding + Standard Scaling).
- Complete model training, evaluation, and visualization charts.
- Interactive Command-Line Interface (`predict.py`).
- Sleek, modern Web Application (`app.py`) built with Flask and Vanilla CSS/JS.
- Automated Unit & Integration test suite (`test_app.py`).

---

## 📊 Model Performance Comparison

| Metric | Linear Regression | Random Forest Regressor | Interpretation |
| :--- | :---: | :---: | :--- |
| **R² Score (Accuracy)** | **90.91%** | **98.15% (Winner)** | Random Forest captures 98.15% of price variance |
| **Mean Absolute Error (MAE)** | ₹ 17.93 Lakhs | **₹ 6.76 Lakhs** | Average error magnitude on unseen test data |
| **Root Mean Squared Error (RMSE)** | ₹ 28.69 Lakhs | **₹ 12.94 Lakhs** | Penalizes larger outlier errors |
| **Mean Absolute % Error (MAPE)** | 19.16% | **5.21%** | Very high reliability and low relative deviation |

---

## 📁 Project Structure

```
d:/3sem/ML projects/house_price_prediction/
│
├── dataset/
│   └── house_data.csv                 # 2,500 real estate records
│
├── models/
│   ├── linear_regression_pipeline.pkl # Trained Linear Regression model
│   ├── random_forest_pipeline.pkl     # Trained Random Forest model
│   └── metadata.json                  # Performance metrics, locations & feature weights
│
├── static/
│   ├── css/
│   │   └── style.css                  # Modern dark glassmorphism design system
│   ├── js/
│   │   └── app.js                     # Dynamic UI interactions & REST API calls
│   └── images/
│       ├── actual_vs_predicted.png    # Scatter plot ground-truth vs predictions
│       ├── feature_importance.png     # Feature importance bar chart
│       └── model_comparison.png       # Metrics comparison chart
│
├── templates/
│   └── index.html                     # Responsive web app interface
│
├── generate_dataset.py                # Generates realistic real estate dataset
├── train_model.py                     # Trains & evaluates both ML models
├── predict.py                         # Single & batch inference + Interactive CLI
├── app.py                             # Flask web server & REST API
├── test_app.py                        # Automated test suite
├── run_project.bat                    # 1-Click launcher for Windows
└── README.md                          # Documentation & viva notes
```

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web App (Recommended)
Open your terminal or command prompt in the project folder and run:
```bash
python app.py
```
Then open your browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

### 2. Run Interactive Terminal Prediction (CLI)
To test predictions directly from your terminal:
```bash
python predict.py
```

### 3. Re-train Machine Learning Models
To train the models from scratch and regenerate comparison plots:
```bash
python train_model.py
```

### 4. Run Automated Test Suite
To verify that all models, APIs, and pages are functioning correctly:
```bash
python test_app.py
```

---

## 🧠 Machine Learning Concepts & Viva Questions

### Q1: Why did Random Forest perform better than Linear Regression?
- **Linear Regression** assumes a strictly linear additive relationship between features and target ($y = mx + c$).
- In real estate, large luxury penthouses have non-linear price jumps, and specific locations (e.g. Koramangala vs Electronic City) interact non-linearly with area and age.
- **Random Forest** builds 150 decision trees using bootstrap aggregating (bagging) with random feature subsets. It easily captures non-linear interactions and splits, leading to an **R² of 98.15%**.

### Q2: What is the meaning of R² Score (Coefficient of Determination)?
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
An $R^2$ of **0.98** means that 98% of the variability in house prices is explained by the model's features (Area, Location, BHK, etc.), while only 2% remains unexplained.

### Q3: How did you handle categorical features like Location?
We utilized `OneHotEncoder(handle_unknown='ignore')` inside a Scikit-Learn `ColumnTransformer`. This converts categorical location names into binary dummy columns ($loc\_Indiranagar$, $loc\_Whitefield$, etc.) without imposing any false ordinal ranking.

### Q4: Which feature has the highest importance?
According to Random Forest Gini/MDI Feature Importance:
1. **Total Built-up Area (sq. ft.):** ~62.5% importance
2. **Location Premium (Koramangala, Indiranagar, etc.):** ~28.0% importance
3. **Property Age & Bathrooms:** ~9.5% importance
