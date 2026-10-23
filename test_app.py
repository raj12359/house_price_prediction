"""
Automated Integration and Unit Tests for House Price Prediction System
"""

import unittest
import json
from app import app
from predict import predict_house_price

class HousePricePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True

    def test_direct_prediction(self):
        result = predict_house_price(
            location="Whitefield",
            total_sqft=1200,
            bhk=2,
            bathrooms=2,
            property_age=3,
            parking=1
        )
        self.assertIn("predictions", result)
        rf = result["predictions"]["random_forest"]
        lr = result["predictions"]["linear_regression"]
        
        self.assertGreater(rf["price_lakhs"], 40.0)
        self.assertGreater(lr["price_lakhs"], 40.0)
        self.assertGreater(result["predictions"]["price_per_sqft"], 3000.0)
        print(f"[TEST PASS] Direct prediction for Whitefield 2BHK: RF={rf['formatted_price']}, LR={lr['formatted_price']}")

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"EstateIQ", response.data)
        print("[TEST PASS] GET / successfully rendered index.html")

    def test_metadata_api(self):
        response = self.client.get("/api/metadata")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("locations", data)
        self.assertIn("metrics", data)
        self.assertIn("random_forest", data["metrics"])
        self.assertIn("linear_regression", data["metrics"])
        print(f"[TEST PASS] GET /api/metadata returned {len(data['locations'])} locations and metrics")

    def test_predict_api(self):
        payload = {
            "location": "Koramangala",
            "total_sqft": 1800,
            "bhk": 3,
            "bathrooms": 3,
            "property_age": 1,
            "parking": 1
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        rf_price = data["data"]["predictions"]["random_forest"]["price_lakhs"]
        print(f"[TEST PASS] POST /api/predict for Koramangala 3BHK: Rs. {rf_price} Lakhs")

    def test_static_plot_access(self):
        response = self.client.get("/static/images/actual_vs_predicted.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "image/png")
        print("[TEST PASS] Static image actual_vs_predicted.png is accessible")

if __name__ == "__main__":
    unittest.main()
