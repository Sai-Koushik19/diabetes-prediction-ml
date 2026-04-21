import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# 1. Load Dataset
# ---------------------------
data = pd.read_csv("diabetes.csv")

print("\nFirst 5 rows of dataset:\n")
print(data.head())

# ---------------------------
# 2. Separate Features & Target
# ---------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Train Model
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 5. Predictions
# ---------------------------
predictions = model.predict(X_test)

# ---------------------------
# 6. Model Accuracy
# ---------------------------
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ---------------------------
# 7. Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 8. Example Prediction
# ---------------------------
print("\nExample Prediction:")

input_data = pd.DataFrame(
    [[6,148,72,35,0,33.6,0.627,50]],
    columns=X.columns
)

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Result: Person is likely Diabetic")
else:
    print("Result: Person is not Diabetic")