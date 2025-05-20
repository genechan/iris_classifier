from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different C values
c_values = [0.01, 0.1, 1, 10, 100]
best_accuracy = 0
best_c = 0
best_model = None

for c in c_values:
    model = LogisticRegression(C=c, max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"C={c}: Accuracy = {accuracy:.2f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c = c
        best_model = model

print(f"\nBest C: {best_c} with accuracy: {best_accuracy:.2f}")

# Save the best model
joblib.dump(best_model, 'iris_model.pkl')
print("Best model saved as 'iris_model.pkl'")