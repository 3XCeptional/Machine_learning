You want to dive deeper into understanding categorical variables? Check out the detailed explanation [here](./understanding_categorical_variables.md).

---

Here is the latest file content as of (2025-02-11T01:15:17.896Z):

```py
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```

```sh
Categorical variables:
['Type', 'Method', 'Regionname']
```
---
```python
def score_dataset(X_train,X_valid,y_train,y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

**Score from Approach 2 (Ordinal Encoding)**:

Scikit-learn has a OrdinalEncoder class that can be used to get ordinal encodings. We loop over the categorical variables and apply the ordinal encoder separately to each column.
```py
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

## Output
MAE from Approach 2 (Ordinal Encoding):
165936.40548390493
```

## Sources 

- https://www.kaggle.com/code/alexisbcook/categorical-variables