import os
import pandas as pd

print("Current folder:", os.getcwd())
print("Files in dataset folder:", os.listdir("dataset"))

data = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset loaded successfully")
print(data.head())
