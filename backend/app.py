from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv(r"c:\Users\hp\Downloads\house_segmentation_kmeans (1).csv")

# Features
X = df.drop("HouseID", axis=1)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means model
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init="auto"
)

model.fit(X_scaled)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    area = float(data["area"])
    bedrooms = float(data["bedrooms"])
    bathrooms = float(data["bathrooms"])
    price = float(data["price"])
    age = float(data["age"])

    user_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, price, age]],
        columns=[
            "Area_sqft",
            "Bedrooms",
            "Bathrooms",
            "Price_Lakhs",
            "Age_Years"
        ]
    )

    user_scaled = scaler.transform(user_data)

    cluster = model.predict(user_scaled)

    return jsonify({
        "cluster": int(cluster[0])
    })


if __name__ == "__main__":
    app.run(debug=True)