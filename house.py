import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dataset
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
    n_init=10
)

# Train model
model.fit(X_scaled)

# Save model and scaler
pickle.dump(model, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# -------------------------
# User Input
# -------------------------

area = float(input("Enter Area (sqft): "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
price = float(input("Enter Price (Lakhs): "))
age = float(input("Enter Age (Years): "))

user_data = pd.DataFrame(
    [[area, bedrooms, bathrooms, price, age]],
    columns=["Area_sqft", "Bedrooms", "Bathrooms", "Price_Lakhs", "Age_Years"]
)

# Scale user input
user_scaled = scaler.transform(user_data)

# Predict cluster
cluster = model.predict(user_scaled)

print("House belongs to Cluster:", cluster[0])