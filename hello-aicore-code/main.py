# Load Datasets
from sklearn import datasets
data_house = datasets.fetch_california_housing(data_home='/app/src')
X = data_house['data']
y = data_house['target']

# Partition into Train and test dataset
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
#
# Init model
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
#
# Train model
clf.fit(train_x, train_y)
#
# Test model
test_r2_score = clf.score(test_x, test_y)
# Output will be available in logs of SAP AI Core.
# Not the ideal way of storing /reporting metrics in SAP AI Core, but that is not the focus this tutorial
print(f"Test Data Score {test_r2_score}")

# # Load Datasets
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# import joblib  # To save the model

# # Load California housing dataset
# data_house = datasets.fetch_california_housing(data_home='/app/src')
# X = data_house['data']
# y = data_house['target']

# # Partition into Train and test dataset
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# # Init model
# clf = DecisionTreeRegressor()

# # Train model
# clf.fit(train_x, train_y)

# # Save the trained model to file
# joblib.dump(clf, 'model.joblib')

# # Test model
# test_r2_score = clf.score(test_x, test_y)

# # Print score to logs (can be viewed in AI Launchpad logs)
# print(f"Test Data Score: {test_r2_score}")
# print("Model saved as model.joblib")
