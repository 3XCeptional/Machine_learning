# K-Nearest Neighbors (KNN)

**What is it?**

K-Nearest Neighbors (KNN) is like deciding something based on what your closest buddies are doing! Imagine you're trying to guess if a new song is pop or rock. KNN looks at the songs that are "closest" to it in terms of musical style (its nearest neighbors) and guesses based on what type those neighbors are. It's a simple and intuitive algorithm for both classification and regression.

**Simple Example:**

Suppose you want to classify a new fruit as either an apple or a banana based on its color and size. KNN would look at the 'k' (say, 3) fruits closest to it in color and size and classify the new fruit based on what the majority of those neighbors are.

**Code Snippet (Python with scikit-learn):**

```python
from sklearn.neighbors import KNeighborsClassifier

# Sample data (features, label)
X = [[2, 2], [3, 2], [1, 4], [5, 6], [6, 5], [4, 4]]
y = [0, 0, 1, 1, 1, 0] # 0 = class 1, 1 = class 2

# Train the KNN classifier
model = KNeighborsClassifier(n_neighbors=3) # You can choose the number of neighbors
model.fit(X, y)

# Predict the class for a new data point [3.5, 4]
prediction = model.predict([[3.5, 4]])
print(f"Predicted class: {prediction[0]}")
```

**Suggestion:**

[Next Page Placeholder]

---

Author: 3XCeptional