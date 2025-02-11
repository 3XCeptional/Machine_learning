# K-Nearest Neighbors (KNN)

**What is K-Nearest Neighbors (KNN)?**

K-Nearest Neighbors (KNN) is one of the simplest yet surprisingly effective machine learning algorithms. It's all about **learning from your neighbors**!  Imagine you're trying to classify a new data point. KNN looks at the 'K' closest data points (neighbors) that are already labeled and makes a decision based on the majority class among those neighbors.

Here’s the breakdown:

*   **Lazy Learning:** KNN is a "lazy learner" because it doesn't learn a model explicitly during training. Instead, it memorizes the training data and uses it directly during prediction.
*   **Distance Metric:**  "Closest" is defined by a distance metric, like Euclidean distance (straight-line distance) or Manhattan distance (city block distance). The choice of metric can impact performance.
*   **Choosing K:** The 'K' in KNN is crucial. It's the number of neighbors you consider.
    *   **Small K (e.g., K=1):** Can be sensitive to noise and outliers in the data.
    *   **Large K:**  Can smooth out noise but might miss local patterns. Choosing the right K often involves experimentation.

KNN is intuitive and versatile, working for both classification and regression tasks. It's particularly useful when decision boundaries are irregular and complex.

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