import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Sample dataset
# 0 = fail, 1 = pass
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['Hours']]         # Feature (2D)
y = df['Passed']          # Target (1D)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model training
model = LogisticRegression()
model.fit(X_train, y_train)
t

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Visualization
# Generate probabilities for plotting
x_range = np.linspace(0, 11, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_range, y_prob, color='red', label='Logistic Curve')
plt.title("Logistic Regression: Hours Studied vs Pass Probability")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid(True)
plt.show()
#print(model.predict[])
print(model.predict([[6]]))
# Output: [1]

print(model.predict_proba([[6]]))
# Output: [[0.30, 0.70]]
