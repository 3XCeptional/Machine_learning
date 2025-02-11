# K-Means Clustering

**What is it?**

K-Means Clustering is like automatically sorting a pile of unsorted laundry into groups! Imagine you have a bunch of data points, and you want to find natural groups or clusters within them without knowing what those groups are beforehand. K-Means is a popular algorithm that helps you do just that. It's an unsupervised learning technique used for clustering data.

**Simple Example:**

Think about grouping customers based on their shopping behavior. K-Means can take customer data (like purchase history, age, income) and automatically group them into distinct segments, like "budget shoppers," "luxury buyers," etc.

**Code Snippet (Python with scikit-learn):**

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample data (features)
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 7], [9, 8]])

# Train the K-Means clustering model
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto') # Specify the number of clusters
kmeans.fit(X)

# Get cluster labels and cluster centers
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids:", centroids)
```

**Suggestion:**

[Next Page Placeholder]

---

Author: 3XCeptional