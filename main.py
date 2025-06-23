# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# import joblib

# # Load California housing dataset
# data_house = datasets.fetch_california_housing(data_home='/app/src')
# X = data_house['data']
# y = data_house['target']

# # Split data
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# # Train model
# clf = DecisionTreeRegressor()
# clf.fit(train_x, train_y)

# # Save trained model
# joblib.dump(clf, 'inference/model.joblib')

# # Print R2 score
# test_score = clf.score(test_x, test_y)
# print(f"Test Data Score: {test_score}")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load dataset from public URL
url = "https://raw.githubusercontent.com/sap-tutorials/Tutorials/master/tutorials/ai-core-data/train.csv"
df = pd.read_csv(url)

# Print to verify
print("Columns in dataset:", df.columns.tolist())

# Separate features and target
X = df.drop(columns=["target"])   # Features
y = df["target"]                  # Target column

# Partition into train and test sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# Initialize model
clf = DecisionTreeRegressor()

# Train model
clf.fit(train_x, train_y)

# Evaluate model
test_r2_score = clf.score(test_x, test_y)

# Save model to file
joblib.dump(clf, "house_price_model.joblib")

# Output result (will appear in SAP AI Core logs)
print(f"Test Score: {test_r2_score}")
print("Model saved as house_price_model.joblib")
