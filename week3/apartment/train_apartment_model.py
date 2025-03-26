import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pickle

# Load dataset
data = pd.read_csv("bfs_municipality_and_tax_data.csv")

# Clean and convert tax_income: remove apostrophes and convert to float
data["tax_income"] = data["tax_income"].astype(str).str.replace("'", "", regex=False)
data["tax_income"] = pd.to_numeric(data["tax_income"], errors="coerce")

# Convert other relevant columns to numeric
data["pop"] = pd.to_numeric(data["pop"], errors="coerce")
data["pop_dens"] = pd.to_numeric(data["pop_dens"], errors="coerce")

# Keep only necessary columns and drop rows with missing values
data = data[["pop", "tax_income", "pop_dens"]].dropna()

# Check how many rows are valid
print("Number of valid rows:", len(data))
if data.empty:
    raise ValueError("No valid data left after cleaning. Please check your dataset.")

# Feature Engineering: create tax income per person
data["tax_per_person"] = data["tax_income"] / data["pop"]

# Define input features and target
X = data[["pop", "tax_income", "tax_per_person"]]
y = data["pop_dens"]  # Use a real price column if available instead

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

# Save the trained model
with open("apartment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("apartment_model.pkl created successfully!")
