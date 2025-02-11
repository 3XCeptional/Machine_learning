---
title: ' What‌ ‌are‌ ‌the‌ ‌10 ‌Popular‌ ‌Machine‌ ‌Learning Algorithms?‌'
updated: 2024-11-01 10:30:17Z
created: 2024-09-17 11:11:47Z
latitude: 19.07598370
longitude: 72.87765590
altitude: 0.0000
completed?: no
---

Here’s a set of clear, concise notes on each topic with easy-to-understand explanations and examples:

---

### **1. Linear Regression**

[Dive deeper into Linear Regression](./understanding_linear_regression.md)

- **Purpose**: Predicts a continuous output (like prices, temperatures) based on linear relationships between input and output variables.
- **How it Works**: Fits a straight line (best fit) to minimize the difference between predicted and actual values. It’s represented as \( Y = mX + c \) where \( m \) is the slope and \( c \) is the intercept.
- **Example**: Predicting house prices based on factors like square footage and number of bedrooms.
- **Key Term**: **Mean Squared Error (MSE)** – Measures the average squared difference between predicted and actual values.

---

### **2. Logistic Regression**

- **Purpose**: Used for binary classification (e.g., Yes/No, 0/1 outcomes).
- **How it Works**: Uses the logistic function (S-shaped curve) to estimate probabilities that a data point belongs to a particular class.
- **Example**: Predicting if an email is spam or not (1 = spam, 0 = not spam).
- **Key Term**: **Sigmoid Function** – Converts linear predictions into probabilities, helping in classification.

[Dive deeper into Logistic Regression](./understanding_logistic_regression.md)

---

### **3. SVM (Support Vector Machine)**

- **Purpose**: Classification technique that separates classes with a clear boundary.
- **How it Works**: Creates a hyperplane (line, plane, or higher-dimensional equivalent) that maximally separates the classes with the largest margin.
- **Example**: Classifying if a tumor is malignant or benign based on features like size and texture.
- **Key Term**: **Support Vectors** – Data points that are closest to the decision boundary and influence its position.

[Dive deeper into Support Vector Machines (SVM)](./understanding_support_vector_machines.md)

---

### **4. KNN (K-Nearest Neighbour)**

- **Purpose**: Classification and regression method that categorizes data based on its nearest neighbors.
- **How it Works**: For a given point, it looks at the closest \( k \) points (neighbors) and assigns the majority class or takes the average (in regression).
- **Example**: Recommending movies by finding movies that similar users have liked.
- **Key Term**: **Distance Metrics** – Measures (like Euclidean distance) used to find closest neighbors.

---

### **5. Decision Tree**

- **Purpose**: A versatile model for classification and regression tasks, easy to interpret.
- **How it Works**: Splits data into branches based on features, forming a tree where each node represents a decision.
- **Example**: Deciding whether a patient has a certain disease based on symptoms and lab tests.
- **Key Term**: **Entropy** – Measures the impurity in a dataset, helping in deciding where to split data.

---

### **6. Random Forest**

- **Purpose**: Improves accuracy and stability over single decision trees by averaging results.
- **How it Works**: Uses multiple decision trees (a "forest") built from random subsets of data and averages their predictions.
- **Example**: Predicting loan defaults by analyzing multiple decision trees trained on borrower characteristics.
- **Key Term**: **Bagging (Bootstrap Aggregating)** – Technique of training each tree on a random sample of data to reduce variance.

---

### **7. Naive Bayes**

- **Purpose**: A simple but effective probabilistic classifier based on Bayes’ theorem, often used in text classification.
- **How it Works**: Assumes that features are independent (naive assumption) and calculates the probability of a class given the features.
- **Example**: Classifying if a review is positive or negative based on word frequencies.
- **Key Term**: **Conditional Probability** – Probability of an event occurring given that another event has already occurred.

---

### **8. PCA (Principal Component Analysis)**

- **Purpose**: Dimensionality reduction technique used to simplify datasets while retaining essential information.
- **How it Works**: Transforms original features into a smaller set of uncorrelated variables (principal components) that capture the most variance.
- **Example**: Reducing a dataset with 100 features to a smaller set that retains most of the information, like compressing image data.
- **Key Term**: **Variance** – Measure of how much a feature contributes to distinguishing the data.

---

### **9. Apriori Algorithm**

- **Purpose**: Used for association rule mining, identifying frequent patterns or itemsets in data.
- **How it Works**: Finds frequent itemsets by building up from individual items and uses these to generate rules.
- **Example**: Discovering that customers who buy bread often also buy milk, useful for market basket analysis.
- **Key Term**: **Support** – Measure of how frequently an itemset appears in the data.

---

### **10. K-Means Clustering**

- **Purpose**: Unsupervised learning algorithm for grouping data points into \( k \) clusters based on similarity.
- **How it Works**: Assigns each data point to the nearest cluster center (centroid) and iteratively adjusts centroids until stable.
- **Example**: Segmenting customers into groups based on purchasing behavior.
- **Key Term**: **Centroid** – The center of a cluster, calculated as the mean of all points in the cluster.

---

These notes should make the concepts clearer and provide a quick reference for each algorithm’s purpose, operation, examples, and key terms.
