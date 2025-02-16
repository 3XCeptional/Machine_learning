### Supervised Machine Learning Algorithms

**Supervised learning** is like learning with a teacher. You give the algorithm labeled examples (input-output pairs), and it learns to predict outputs for new, unseen inputs based on this labeled data. It's used for tasks like classification and regression, where the goal is to predict a category or a continuous value. Common supervised algorithms include:

1. **[Linear Regression](Supervised_vs_Unsupervised/supervised/linear regression.md)**
2. **[Logistic Regression](Supervised_vs_Unsupervised/supervised/logistic_regression.md)**
3. **[Support Vector Machines (SVM)](Supervised_vs_Unsupervised/supervised/support_vector_machines.md)**
4. **[Decision Trees](intro%20to%20Machine%20Learning/DecisionTree_and_%20DecisionTreeRegressor.md)**
5. **[Random Forests](Supervised_vs_Unsupervised/supervised/random_forests.md)**
6. **[K-Nearest Neighbors (KNN)](Supervised_vs_Unsupervised/supervised/k_nearest_neighbors.md)**
7. **Naive Bayes**
8. **Gradient Boosting Machines (GBM)**
   - **XGBoost**
   - **LightGBM**
   - **CatBoost**
9. **Artificial Neural Networks (ANN)**
10. **Ridge Regression**
11. **Lasso Regression**

### Unsupervised Machine Learning Algorithms

**Unsupervised learning** is like exploring without a map. You give the algorithm unlabeled data, and it tries to find hidden patterns, structures, or groupings on its own. It's used for tasks like clustering, dimensionality reduction, and anomaly detection, where the goal is to understand the inherent structure of the data. Common unsupervised algorithms include:

1. **[K-Means Clustering](Supervised_vs_Unsupervised/Unsupervised/kmeans.md)**
2. **Hierarchical Clustering**
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
4. **Principal Component Analysis (PCA)**
5. **Independent Component Analysis (ICA)**
6. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
7. **Autoencoders (can be used for unsupervised feature learning)**
8. **Gaussian Mixture Models (GMM)**
9. **Apriori Algorithm (for Association Rule Learning)**
10. **Self-Organizing Maps (SOM)**

### Deep Learning

**Deep learning** is a more advanced area within machine learning, inspired by the structure of the human brain. It uses deep neural networks with many layers to learn very complex patterns from vast amounts of data. Deep learning excels in tasks like image and speech recognition, natural language processing, and complex data analysis. Common deep learning architectures include:

- **Artificial Neural Networks (ANN)**
- **Convolutional Neural Networks (CNN)**
- **Recurrent Neural Networks (RNN)**
  - **Long Short-Term Memory (LSTM)**
  - **Gated Recurrent Units (GRU)**
- **Autoencoders**
- **Deep Belief Networks (DBN)**
- **Generative Adversarial Networks (GANs)**
- **Transformer Networks**

### Common Deep Learning Algorithms

These are some fundamental algorithms and techniques that are commonly used in the context of deep learning:

1. **Backpropagation (for training neural networks)**
2. **Stochastic Gradient Descent (SGD)**
3. **Adam Optimizer**
4. **Dropout (regularization technique)**
5. **Batch Normalization**
6. **Softmax (for classification tasks)**
7. **Activation Functions**
   - **ReLU (Rectified Linear Unit)**
   - **Sigmoid**
   - **Tanh**
   - **Leaky ReLU**



# Supervised Machine Learning Algorithms

Supervised learning involves training a model on labeled data, where the input data and corresponding output labels are provided. The algorithm learns to map inputs to outputs and makes predictions on unseen data. Common supervised algorithms include:

1. **Linear Regression**
2. **Logistic Regression**
3. **Support Vector Machines (SVM)**
4. **Decision Trees**
5. **Random Forests**
6. **K-Nearest Neighbors (KNN)**
7. **Naive Bayes**
8. **Gradient Boosting Machines (GBM)**
   - **XGBoost**
   - **LightGBM**
   - **CatBoost**
9. **Artificial Neural Networks (ANN)**
10. **Ridge Regression**
11. **Lasso Regression**


---

### 1. **Linear Regression**

- **Definition**: Linear regression is a supervised learning algorithm used for predicting a continuous target variable by fitting a straight line (the regression line) through the data points. The line represents the relationship between the independent (input) and dependent (output) variables.
- **Example**: Predicting house prices based on features like size, number of rooms, and location.

---

### 2. **Logistic Regression**

- **Definition**: Logistic regression is a supervised learning algorithm used for binary classification tasks. It predicts the probability of a categorical outcome (often 0 or 1) using the logistic function (sigmoid curve) to map any real-valued number into the range [0,1].
- **Example**: Classifying whether an email is spam or not based on the content of the email.

---

### 3. **Support Vector Machines (SVM)**

- **Definition**: SVM is a supervised learning algorithm used for classification tasks. It works by finding the hyperplane that best separates the data points of different classes with the maximum margin between them.
- **Example**: Classifying handwritten digits (0–9) based on pixel intensity features.

---

### 4. **Decision Trees**

- **Definition**: Decision trees are supervised learning algorithms used for both classification and regression tasks. They split the data into branches at decision nodes based on feature values, forming a tree-like structure. Each branch leads to a decision or classification.
- **Example**: Predicting whether a customer will buy a product based on features like age, income, and browsing behavior.

---

### 5. **Random Forests**

- **Definition**: Random forests are an ensemble learning method that combines multiple decision trees to improve accuracy. It aggregates the predictions from individual trees to make a final decision, which reduces overfitting and improves generalization.
- **Example**: Predicting credit card fraud by aggregating predictions from multiple decision trees built from transaction data.

---

### 6. **K-Nearest Neighbors (KNN)**

- **Definition**: KNN is a simple supervised learning algorithm used for both classification and regression. It classifies a data point based on the majority class of its 'k' nearest neighbors or predicts the average value for regression.
- **Example**: Classifying whether a person has diabetes based on the health metrics of their nearest neighbors in the dataset.

---

### 7. **Naive Bayes**

- **Definition**: Naive Bayes is a probabilistic supervised learning algorithm based on Bayes' theorem. It assumes that the features are conditionally independent given the class label. Despite the 'naive' assumption, it often performs well on large datasets.
- **Example**: Classifying text documents (e.g., spam detection) based on the frequency of words in the document.

---

### 8. **Gradient Boosting Machines (GBM)**

- **Definition**: GBM is an ensemble learning technique that builds multiple weak models (usually decision trees) sequentially, where each model tries to correct the errors of the previous one. The goal is to minimize the loss function by adding models that correct mistakes.
- **Example**: Predicting customer churn based on past behavior, purchase history, and customer service interactions.

---

### 9. **XGBoost**

- **Definition**: XGBoost (Extreme Gradient Boosting) is an optimized version of GBM. It is known for its speed and performance due to techniques like parallelization, regularization, and efficient memory usage. It works well for structured/tabular data.
- **Example**: Winning entries in data science competitions like Kaggle often use XGBoost for problems such as predicting sales revenue based on historical sales data.

---

### 10. **LightGBM**

- **Definition**: LightGBM is another gradient boosting algorithm designed to be even faster than XGBoost. It uses a novel technique called leaf-wise growth for constructing decision trees, which makes it highly efficient for large datasets.
- **Example**: Predicting loan default based on customer demographic data and transaction history in financial datasets.

---

### 11. **CatBoost**

- **Definition**: CatBoost is a gradient boosting algorithm that handles categorical variables automatically without requiring extensive preprocessing. It is fast and achieves high performance, especially on datasets with categorical features.
- **Example**: Classifying customer behavior in e-commerce platforms where customer attributes like city, product type, and purchase frequency are categorical.

---

### 12. **Artificial Neural Networks (ANN)**

- **Definition**: ANNs are a class of supervised learning algorithms inspired by the structure of the human brain. They consist of layers of interconnected neurons that process data by adjusting weights and biases during training. ANNs are commonly used for complex tasks like image and speech recognition.
- **Example**: Recognizing handwritten digits from images using a neural network.

---

### 13. **Ridge Regression**

- **Definition**: Ridge regression (also known as L2 regularization) is a variant of linear regression that adds a penalty term to the loss function. This penalty discourages the model from having overly complex or large coefficients, helping to reduce overfitting.
- **Example**: Predicting the price of a car based on features like engine size, mileage, and age, while avoiding overfitting the model by shrinking coefficients.

---

### 14. **Lasso Regression**

- **Definition**: Lasso regression (also known as L1 regularization) is similar to ridge regression but uses a penalty that can shrink some coefficients to zero. This leads to a sparse model, which can be useful for feature selection.
- **Example**: Predicting housing prices by selecting the most relevant features (e.g., number of bedrooms, neighborhood) while ignoring less important ones.
