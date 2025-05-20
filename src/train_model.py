from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset (like a table of flower measurements)
iris = load_iris()
X = iris.data  # Measurements: sepal length, sepal width, petal length, petal width
y = iris.target  # Numbers 0, 1, 2 for setosa, versicolor, virginica

# Make a table to view the data
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# Show some info
print("Dataset size:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nHow many of each species:\n", df['species'].value_counts())