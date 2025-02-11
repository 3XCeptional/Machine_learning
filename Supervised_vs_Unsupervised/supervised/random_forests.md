# Random Forests

**What is it?**

Random Forests are like a team of decision-making trees working together! Imagine you're trying to predict if it will rain. Instead of asking just one weather expert (a single decision tree), you ask a whole bunch of them, and they each give their opinion. Random Forests combine all these opinions to make a super prediction that's usually more accurate and reliable than just one tree. It's a powerful ensemble method for both classification and regression.

**Simple Example:**

Let's say we want to predict if a customer will like a movie. We can build a Random Forest model that considers opinions from many decision trees, each looking at different aspects of the movie and customer preferences, to make a final prediction.

**Code Snippet (Python with scikit-learn):**

```python
from sklearn.ensemble import RandomForestClassifier

# Sample data (features, label)
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
y = [0, 0, 0, 1, 1, 1] # 0 = class 1, 1 = class 2

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100) # You can adjust the number of trees
model.fit(X, y)

# Predict the class for a new data point [6, 7]
prediction = model.predict([[6, 7]])
print(f"Predicted class: {prediction[0]}")
```

**Suggestion:**

[Next Page Placeholder]

---

Author: 3XCeptional