import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Separate input features and output label
X = data.drop("label", axis=1)
y = data["label"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("model/crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model/crop_model.pkl")
