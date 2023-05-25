import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Step 2: Preprocess the data
X = X / 255.0  # Normalize pixel values to the range [0, 1]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the model
model = LogisticRegression(solver='saga', max_iter=100)

# Step 5: Implement the Gradient Descent algorithm
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)