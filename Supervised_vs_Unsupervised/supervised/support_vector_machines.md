# Support Vector Machines (SVM)

**What is it?**

Support Vector Machines (SVM) are like the ultimate boundary creators! Imagine you have two groups of things (like cats and dogs) and you want to draw a line to separate them as cleanly as possible. SVM does exactly that, but in a super smart way to handle even tricky situations. It's excellent for classification tasks.

**Simple Example:**

Think about sorting apples and oranges. SVM can learn to draw the best line (or even curve!) to separate apples from oranges based on their features like size and color.

**Code Snippet (Python with scikit-learn):**

```python
from sklearn.svm import SVC

# Sample data (feature 1, feature 2, label)
X = [[2, 2], [3, 2], [1, 4], [5, 6], [6, 5], [4, 4]]
y = [0, 0, 1, 1, 1, 0] # 0 = class 1, 1 = class 2

# Train the SVM classifier
model = SVC(kernel='linear') # You can choose different kernels
model.fit(X, y)

# Predict the class for a new data point [3.5, 4]
prediction = model.predict([[3.5, 4]])
print(f"Predicted class: {prediction[0]}")
```

**Suggestion:**

[Next Page Placeholder]

---

Author: 3XCeptional