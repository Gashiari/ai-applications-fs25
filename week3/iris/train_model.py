import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
 
# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
 
# Feature Engineering: New feature (Sepal Ratio)
df["sepal_ratio"] = df["sepal length (cm)"] / df["sepal width (cm)"]
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df, iris.target, test_size=0.2, random_state=42
)
 
# Train the improved RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate model performance
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Save the trained model to file
with open("improved_iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
 
print("Model trained and saved successfully as 'improved_iris_model.pkl'.")