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

**Simple Example: Movie Recommendation**

Imagine you're building a movie recommendation system. KNN can help suggest movies a user might like based on the preferences of similar users!

Let's say we have data on users and their movie ratings.  For a new user, we want to recommend movies. KNN can work like this:

1.  **Find similar users:**  We use KNN to find the 'K' users who are most similar to the new user based on their movie ratings. Similarity can be measured by distance metrics on user-rating vectors.
2.  **Aggregate neighbors' preferences:** Look at the movies those 'K' nearest neighbors have liked (rated highly).
3.  **Recommend top movies:** Recommend the movies that are popular among the neighbors but haven't been seen by the new user yet.

For instance, if you like action and sci-fi movies, KNN would find other users with similar taste profiles and recommend movies that those users have enjoyed, but you haven't watched yet.

This example illustrates how KNN can be used for recommendation systems by finding similar entities (users in this case) and leveraging their preferences to make predictions for a new entity.

**Code Snippet: Movie Recommendation with KNN in scikit-learn**

```python
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Sample user-movie ratings data (users as rows, movies as columns, ratings 1-5)
data = {'User1': [4, 5, 3, None, 2],
        'User2': [5, None, 4, 5, 3],
        'User3': [3, 4, None, 4, 5],
        'User4': [None, 5, 3, 4, 2],
        'User5': [2, 3, 5, None, 4]}
movies = ['MovieA', 'MovieB', 'MovieC', 'MovieD', 'MovieE']
df = pd.DataFrame(data, index=movies).T # Transpose for user-rows, movie-columns

user_ratings = df.fillna(0) # Fill NaN with 0 for simplicity

# Example: Recommend movies for User 'User1'

user_index = 0 # Index for 'User1'
user_profile = user_ratings.iloc[user_index].values.reshape(1, -1)

# Train KNN classifier - using cosine distance for user similarity
knn_model = KNeighborsClassifier(n_neighbors=3, metric='cosine') # Experiment with different 'k' and metrics
knn_model.fit(user_ratings, range(len(user_ratings))) # Target is user index for simplicity


# Find neighbors (similar users)
distances, neighbor_indices = knn_model.kneighbors(user_profile)

print(f"Recommendations for User1 based on K=3 (Cosine Distance):")
for neighbor_index in neighbor_indices[0]:
    if neighbor_index != user_index: # Exclude self
        neighbor_name = user_ratings.index[neighbor_index]
        recommendations = df.loc[df.loc[neighbor_name] >= 4, movies].index.tolist() # Movies neighbor liked (rated 4+)
        print(f"- From similar user '{neighbor_name}': Recommend movies: {recommendations}")


# Example with different K and metric (Manhattan Distance)
knn_model_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_model_manhattan.fit(user_ratings, range(len(user_ratings)))
distances_manhattan, neighbor_indices_manhattan = knn_model_manhattan.kneighbors(user_profile)

print(f"\nRecommendations for User1 based on K=5 (Manhattan Distance):")
for neighbor_index in neighbor_indices_manhattan[0]:
    if neighbor_index != user_index: # Exclude self
        neighbor_name = user_ratings.index[neighbor_index]
        recommendations = df.loc[df.loc[neighbor_name] >= 4, movies].index.tolist() # Movies neighbor liked (rated 4+)
        print(f"- From similar user '{neighbor_name}': Recommend movies from '{neighbor_name}': {recommendations}")

```

**Explanation of code enhancements:**

1.  **User-Movie Rating Data:**  Uses a Pandas DataFrame to represent user-movie ratings, a common format for recommendation systems.
2.  **Cosine Distance:** Demonstrates using 'cosine' distance, which is often more suitable for user similarity based on ratings than Euclidean distance.
3.  **Impact of K:** Shows how changing 'K' (number of neighbors) can affect recommendations.
4.  **Impact of Distance Metric:**  Illustrates how different distance metrics ('cosine' vs. 'manhattan') can lead to different neighbor sets and recommendations.
5.  **Recommendation Logic:**  Provides basic recommendation logic: find similar users, recommend movies they liked.

This enhanced code snippet provides a more relevant and insightful example of using KNN for movie recommendations and highlights the importance of choosing appropriate 'K' values and distance metrics in KNN-based recommendation systems.

**Suggestion:**

Next up: [K-Means Clustering](Supervised_vs_Unsupervised/Unsupervised/k_means_clustering.md)

---

Author: 3XCeptional