# LEGO Regression Project

## Introduction 

In this cumulative lab you'll perform a full linear regression analysis and report the findings of your final model, including both predictive model performance metrics and interpretation of fitted model parameters.

## Objectives

You will be able to:

* Perform a full linear regression analysis with iterative model development
* Evaluate your final model and interpret its predictive performance metrics
* Apply an inferential lens to interpret relationships between variables identified by the model

# Task: Develop a LEGO Pricing Algorithm

![pile of legos](images/legos.jpg)

Photo by <a href="https://unsplash.com/@xavi_cabrera?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Xavi Cabrera</a> on <a href="/s/photos/lego?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Business Understanding

You just got hired by LEGO! Your first project is going to be to develop a pricing algorithm to help set a target price for new LEGO sets that are released to market. The goal is to save the company some time and to help ensure consistency in pricing between new products and past products.

The main purpose of this algorithm is *predictive*, meaning that **your model should be able to take in attributes of a LEGO set that does not yet have a set price, and to predict a good price**. The effectiveness of your predictive model will be measured by how well it predicts prices in our test set, where we know what the actual prices were but the model does not.

The secondary purpose of this algorithm is *inferential*, meaning that **your model should be able to tell us something about the relationship between the attributes of a LEGO set and its price**. You will apply your knowledge of statistics to include appropriate caveats about these relationships.

## Data Understanding

You have access to a dataset containing over 700 LEGO sets released in the past, including attributes of those sets as well as their prices. You can assume that the numeric attributes in this dataset have already been preprocessed appropriately for modeling (i.e. that there are no missing or invalid values), while the text attributes are simply there for your visual inspection and should not be used for modeling. Also, note that some of these attributes cannot be used in your analysis because they will be unavailable for future LEGO products or are otherwise irrelevant.

You do not need to worry about inflation or differences in currency; just predict the same kinds of prices as are present in the past data, which have already been converted to USD.

## Loading the Data

Load both the train and test datasets from the `data` folder in this repository, then split them into feature and target DataFrames. Remember, both of these datasets contain prices, but we are using the test set as a stand-in for future LEGO products where the price has not yet been determined. The model will be trained on just the train set, then we will compare its predictions on the test set to the actual prices on the test set.


```python
# Run this cell without changes
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# __SOLUTION__
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Replace None with your code

train = None
test = None

X_train = None
y_train = None

X_test = None
y_test = None
```


```python
# __SOLUTION__

train = pd.read_csv("data/lego_train.csv")
test = pd.read_csv("data/lego_test.csv")

X_train = train.drop("list_price", axis=1)
y_train = train["list_price"]

X_test = test.drop("list_price", axis=1)
y_test = test["list_price"]
```

## Exploring the Data

Inspect your data to start making sense of it


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>set_name</th>
      <th>prod_desc</th>
      <th>theme_name</th>
      <th>piece_count</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>difficulty_level</th>
      <th>num_reviews</th>
      <th>star_rating</th>
      <th>list_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60123</td>
      <td>Volcano Supply Helicopter</td>
      <td>Fly in the exploration gear and fly out the cr...</td>
      <td>City</td>
      <td>330</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>4.3</td>
      <td>49.990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71246</td>
      <td>Adventure Time™ Team Pack</td>
      <td>Explore the Land of Ooo with Jake and Lumpy Sp...</td>
      <td>DIMENSIONS™</td>
      <td>96</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>4.7</td>
      <td>30.362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10616</td>
      <td>My First Playhouse</td>
      <td>Learn about daily routines with this easy-to-b...</td>
      <td>DUPLO®</td>
      <td>25</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>19.990</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31079</td>
      <td>Sunshine Surfer Van</td>
      <td>Enjoy seaside fun with the 3-in-1 Sunshine Sur...</td>
      <td>Creator 3-in-1</td>
      <td>379</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.4</td>
      <td>34.990</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42057</td>
      <td>Ultralight Helicopter</td>
      <td>Take to the skies with the Ultralight Helicopter!</td>
      <td>Technic</td>
      <td>199</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>9.0</td>
      <td>4.7</td>
      <td>19.990</td>
    </tr>
  </tbody>
</table>
</div>



Produce high-level descriptive information about your training data


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>piece_count</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>difficulty_level</th>
      <th>num_reviews</th>
      <th>star_rating</th>
      <th>list_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.580000e+02</td>
      <td>558.000000</td>
      <td>558.000000</td>
      <td>558.000000</td>
      <td>558.000000</td>
      <td>490.000000</td>
      <td>490.000000</td>
      <td>558.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.575458e+04</td>
      <td>432.806452</td>
      <td>6.928315</td>
      <td>27.577061</td>
      <td>1.514337</td>
      <td>16.857143</td>
      <td>4.516327</td>
      <td>48.645385</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.241344e+05</td>
      <td>730.305859</td>
      <td>2.987146</td>
      <td>33.747031</td>
      <td>0.801015</td>
      <td>36.654636</td>
      <td>0.487826</td>
      <td>71.168371</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.300000e+02</td>
      <td>1.000000</td>
      <td>1.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.600000</td>
      <td>2.490000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.118475e+04</td>
      <td>91.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.300000</td>
      <td>14.990000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.207350e+04</td>
      <td>199.500000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>4.600000</td>
      <td>29.990000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.125325e+04</td>
      <td>467.000000</td>
      <td>8.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>14.000000</td>
      <td>5.000000</td>
      <td>49.990000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000431e+06</td>
      <td>7541.000000</td>
      <td>16.000000</td>
      <td>99.000000</td>
      <td>4.000000</td>
      <td>367.000000</td>
      <td>5.000000</td>
      <td>799.990000</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the distribution of the target variable


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(y_train, bins=100)

ax.set_xlabel("Listing Price (USD)")
ax.set_ylabel("Count of LEGO Sets")
ax.set_title("Distribution of LEGO Set Prices");
```


    
![png](index_files/index_15_0.png)
    


## Requirements

### 1. Interpret a Correlation Heatmap to Build a Baseline Model

You'll start modeling by choosing the feature that is most correlated with our target, and build and evaluate a linear regression model with just that feature.

### 2. Build a Model with All Relevant Numeric Features

Now, add in the rest of the relevant numeric features of the training data, and compare that model's performance to the performance of the baseline model.

### 3. Select the Best Combination of Features

Using statistical properties of the fitted model, the `sklearn.feature_selection` submodule, and some custom code, find the combination of relevant numeric features that produces the best scores.

### 4. Build and Evaluate a Final Predictive Model

Using the best features selected in the previous step, create a final model, fit it on all rows of the training dataset, and evaluate it on all rows of the test dataset in terms of both r-squared and RMSE.

### 5. Interpret the Final Model

Determine what, if any, understanding of the underlying relationship between variables can be determined with this model. This means you will need to interpret the model coefficients as well as checking whether the assumptions of linear regression have been met.

### 6. Create Presentation Notebook

Edit this notebook or create a new one to showcase your work.

# 1. Interpret a Correlation Heatmap to Build a Baseline Model

## Correlation Heatmap

Produce a heatmap showing the correlations between all of the numeric values in our training data. The x and y axis labels should indicate the pair of values that are being compared, and then the color and the number should represent the correlation. 

The most important column or row shows the correlations between the target (listing price) and other attributes.


```python
# Run this cell without changes

import seaborn as sns
import numpy as np
```


```python
# __SOLUTION__

import seaborn as sns
import numpy as np
```


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

# Create a df with the target as the first column,
# then compute the correlation matrix
heatmap_data = train
corr = heatmap_data.corr()

# Set up figure and axes
fig, ax = plt.subplots(figsize=(5, 8))

# Plot a heatmap of the correlation matrix, with both
# numbers and colors indicating the correlations
sns.heatmap(
    # Specifies the data to be plotted
    data=corr,
    # The mask means we only show half the values,
    # instead of showing duplicates. It's optional.
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    # Specifies that we should use the existing axes
    ax=ax,
    # Specifies that we want labels, not just colors
    annot=True,
    # Customizes colorbar appearance
    cbar_kws={"label": "Correlation", "orientation": "horizontal", "pad": .2, "extend": "both"}
)

# Customize the plot appearance
ax.set_title("Heatmap of Correlation Between Attributes (Including Target)");
```


    
![png](index_files/index_21_0.png)
    


Based on the heatmap, which feature is most strongly correlated with the target (`listing_price`)? In other words, which feature has the strongest positive or negative correlation — the correlation with the greatest magnitude?


```python
# Replace None with the name of the feature (a string)

most_correlated_feature = None
```


```python
# __SOLUTION__

most_correlated_feature = "piece_count"
```

Create a scatter plot of that feature vs. listing price:


```python
# Replace None with your code

None
```


```python
# __SOLUTION__

fig, ax = plt.subplots()

ax.scatter(X_train[most_correlated_feature], y_train, alpha=0.5)
ax.set_xlabel(most_correlated_feature)
ax.set_ylabel("listing price")
ax.set_title("Most Correlated Feature vs. Listing Price");
```


    
![png](index_files/index_27_0.png)
    


Assuming you correctly identified `piece_count` (the number of pieces in the LEGO set) as the most correlated feature, you should have a scatter plot that shows a fairly clear linear relationship between that feature and the target. It looks like we are ready to proceed with making our baseline model without any additional transformation.

## Building a Baseline Model

Now, we'll build a linear regression model using just that feature, which will serve as our baseline model:


```python
# Run this cell without changes

from sklearn.linear_model import LinearRegression

baseline_model = LinearRegression()
```


```python
# __SOLUTION__

from sklearn.linear_model import LinearRegression

baseline_model = LinearRegression()
```

Then we evaluate the model using `cross_validate` and `ShuffleSplit`, which essentially means that we perform 3 separate train-test splits within our `X_train` and `y_train`, then we find both the train and the test scores for each.


```python
# Run this cell without changes

from sklearn.model_selection import cross_validate, ShuffleSplit

splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

baseline_scores = cross_validate(
    estimator=baseline_model,
    X=X_train[[most_correlated_feature]],
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```


```python
# __SOLUTION__

from sklearn.model_selection import cross_validate, ShuffleSplit

splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

baseline_scores = cross_validate(
    estimator=baseline_model,
    X=X_train[[most_correlated_feature]],
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```

    Train score:      0.7785726407224942
    Validation score: 0.7793473618106956


Interpret these scores below. What are we measuring? What can we learn from this?

**Hint:** when you use `cross_validate`, it uses the `.score` method of the estimator by default. See [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score) for that method of `LinearRegression`.


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
Because we are using the .score method of LinearRegression, these
are r-squared scores. That means that each of them represents the
amount of variance of the target (listing price) that is explained
by the model's features (currently just the number of pieces) and
parameters (intercept value and coefficient values for the features)

In general this seems like a fairly strong model already. It is
getting nearly identical performance on training subsets compared to
the validation subsets, explaining around 80% of the variance both
times
"""
```




    "\nBecause we are using the .score method of LinearRegression, these\nare r-squared scores. That means that each of them represents the\namount of variance of the target (listing price) that is explained\nby the model's features (currently just the number of pieces) and\nparameters (intercept value and coefficient values for the features)\n\nIn general this seems like a fairly strong model already. It is\ngetting nearly identical performance on training subsets compared to\nthe validation subsets, explaining around 80% of the variance both\ntimes\n"



# 2. Build a Model with All Numeric Features

Now that we have established a baseline, it's time to move on to more complex models.

## Numeric Feature Selection

One thing that you will almost always need to do in a modeling process is remove non-numeric data prior to modeling. While you could apply more-advanced techniques such as one-hot encoding or NLP in order to convert non-numeric columns into numbers, this time just create a dataframe `X_train_numeric` that is a copy of `X_train` that only contains numeric columns.

You can look at the `df.info()` printout above to do this manually, or there is a handy `.select_dtypes` method ([documentation here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)).


```python
# Replace None with appropriate code

X_train_numeric = None

X_train_numeric
```


```python
# __SOLUTION__

X_train_numeric = X_train.select_dtypes("number").copy()

X_train_numeric
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>piece_count</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>difficulty_level</th>
      <th>num_reviews</th>
      <th>star_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60123</td>
      <td>330</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71246</td>
      <td>96</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10616</td>
      <td>25</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31079</td>
      <td>379</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42057</td>
      <td>199</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>9.0</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>553</th>
      <td>71343</td>
      <td>56</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>554</th>
      <td>75114</td>
      <td>81</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>0</td>
      <td>10.0</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>555</th>
      <td>41597</td>
      <td>708</td>
      <td>10.0</td>
      <td>99.0</td>
      <td>2</td>
      <td>13.0</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>556</th>
      <td>75116</td>
      <td>98</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>557</th>
      <td>76097</td>
      <td>406</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>2</td>
      <td>5.0</td>
      <td>4.8</td>
    </tr>
  </tbody>
</table>
<p>558 rows × 7 columns</p>
</div>



The following code checks that your answer was correct:


```python
# Run this cell without changes

# X_train_numeric should be a dataframe
assert type(X_train_numeric) == pd.DataFrame

# Check for correct shape
assert X_train_numeric.shape == (558, 7)
```


```python
# __SOLUTION__

# X_train_numeric should be a dataframe
assert type(X_train_numeric) == pd.DataFrame

# Check for correct shape
assert X_train_numeric.shape == (558, 7)
```

Now we can look at scatter plots of all numeric features compared to the target (skipping `piece_count` since we already looked at that earlier):


```python
# Run this cell without changes

scatterplot_data = X_train_numeric.drop("piece_count", axis=1)

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
fig.set_tight_layout(True)

for index, col in enumerate(scatterplot_data.columns):
    ax = axes[index//3][index%3]
    ax.scatter(X_train_numeric[col], y_train, alpha=0.2)
    ax.set_xlabel(col)
    ax.set_ylabel("listing price")
```


```python
# __SOLUTION__

scatterplot_data = X_train_numeric.drop("piece_count", axis=1)

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
fig.set_tight_layout(True)

for index, col in enumerate(scatterplot_data.columns):
    ax = axes[index//3][index%3]
    ax.scatter(X_train_numeric[col], y_train, alpha=0.2)
    ax.set_xlabel(col)
    ax.set_ylabel("listing price")
```


    
![png](index_files/index_46_0.png)
    


## Feature Selection Using Domain Understanding

Ok, now all of the remaining features can technically go into a model with scikit-learn. But do they make sense?

Some reasons you might not want to include a given numeric column include:

1. The column represents a unique identifier, not an actual numeric feature
2. The column is something that will not be available when making future predictions

Recall that the business purpose here is creating an algorithm to set the price for a newly-released LEGO set. Which columns should we drop because of the issues above?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
The first issue aligns with the first feature we have, prod_id

While it is possible that there is some useful information 
encoded in that number, it seems like it is not really a numeric
feature in the traditional sence

The scatter plot supports this idea, since it shows almost all
prices being represented by a narrow range of ID values

The second issue aligns with num_reviews and star_rating. Although
these might be useful features in some modeling context, they are
not useful for this algorithm because we won't know the number of
reviews or the star rating until after the LEGO set is released.
"""
```




    "\nThe first issue aligns with the first feature we have, prod_id\n\nWhile it is possible that there is some useful information \nencoded in that number, it seems like it is not really a numeric\nfeature in the traditional sence\n\nThe scatter plot supports this idea, since it shows almost all\nprices being represented by a narrow range of ID values\n\nThe second issue aligns with num_reviews and star_rating. Although\nthese might be useful features in some modeling context, they are\nnot useful for this algorithm because we won't know the number of\nreviews or the star rating until after the LEGO set is released.\n"



Now, create a variable `X_train_second_model`, which is a copy of `X_train_numeric` where those irrelevant columns have been removed:


```python
# Replace None with appropriate code

X_train_second_model = None

X_train_second_model
```


```python
# __SOLUTION__

X_train_second_model = X_train_numeric.drop(["prod_id", "num_reviews", "star_rating"], axis=1).copy()

X_train_second_model
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>difficulty_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>330</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>96</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>379</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>199</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>553</th>
      <td>56</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>554</th>
      <td>81</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>555</th>
      <td>708</td>
      <td>10.0</td>
      <td>99.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>556</th>
      <td>98</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>557</th>
      <td>406</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>558 rows × 4 columns</p>
</div>



## Building and Evaluating the Second Model

In the cell below, we use the same process to evaluate a model using `X_train_second_model` rather than using just `piece_count`.


```python
# Run this cell without changes

second_model = LinearRegression()

second_model_scores = cross_validate(
    estimator=second_model,
    X=X_train_second_model,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Current Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```


```python
# __SOLUTION__

second_model = LinearRegression()

second_model_scores = cross_validate(
    estimator=second_model,
    X=X_train_second_model,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Current Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```

    Current Model
    Train score:      0.7884552982196166
    Validation score: 0.7558203636660551
    
    Baseline Model
    Train score:      0.7785726407224942
    Validation score: 0.7793473618106956


Interpret these results. Did our second model perform better than the baseline? Any ideas about why or why not?

**Hint:** because the purpose of this model is to set future prices that have not been determined yet, the most important metric for evaluating model performance is the validation score, not the train score.


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
Our second model got slightly better scores on the training
data, but worse scores on the validation data. This means that
it is a worse model overall, since what we care about is the
ability to generate prices for future LEGO sets, not the
ability to fit well to the known LEGO sets' features

It seems like adding in these other features is actually just
causing overfitting, rather than improving the model's ability
to understand the underlying patterns in the data
"""
```




    "\nOur second model got slightly better scores on the training\ndata, but worse scores on the validation data. This means that\nit is a worse model overall, since what we care about is the\nability to generate prices for future LEGO sets, not the\nability to fit well to the known LEGO sets' features\n\nIt seems like adding in these other features is actually just\ncausing overfitting, rather than improving the model's ability\nto understand the underlying patterns in the data\n"



# 3. Select the Best Combination of Features

As you likely noted above, adding all relevant numeric features did not actually improve the model performance. Instead, it led to overfitting.

## Investigating Multicollinearity

This potentially indicates that our model is performing poorly because these features violate the independence assumption (i.e. there is strong multicollinearity). In other words, maybe the minimum age, maximum age, and difficulty level are not really providing different information than the number of pieces in the LEGO set, and instead are just adding noise. Then the model is using that noise to get a slightly better score on the training data, but but a worse score on the validation data.

While `LinearRegression` from scikit-learn has a lot of nice functionality for working with a predictive framing (e.g. compatibility with the `cross_validate` function), it doesn't have anything built in to detect strong multicollinearity. Fortunately the same linear regression model is also available from StatsModels ([documentation here](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)), where it is called `OLS` (for "ordinary least squares"). Models in StatsModels, including `OLS`, are not really designed for predictive model validation, but they do give us a lot more statistical information.

In the cell below, use StatsModels to fit a linear regression model on the same features used in the second model, then display the summary. 


```python
# Replace None with your code

import statsmodels.api as sm

None
```


```python
# __SOLUTION__
import statsmodels.api as sm

sm.OLS(y_train, sm.add_constant(X_train_second_model)).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>list_price</td>    <th>  R-squared:         </th> <td>   0.786</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.784</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   506.8</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 28 May 2021</td> <th>  Prob (F-statistic):</th> <td>2.40e-183</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:13:18</td>     <th>  Log-Likelihood:    </th> <td> -2741.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   558</td>      <th>  AIC:               </th> <td>   5493.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   553</td>      <th>  BIC:               </th> <td>   5514.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>            <td>   22.0991</td> <td>    4.550</td> <td>    4.857</td> <td> 0.000</td> <td>   13.162</td> <td>   31.037</td>
</tr>
<tr>
  <th>piece_count</th>      <td>    0.0906</td> <td>    0.003</td> <td>   33.407</td> <td> 0.000</td> <td>    0.085</td> <td>    0.096</td>
</tr>
<tr>
  <th>min_age</th>          <td>   -2.7627</td> <td>    0.696</td> <td>   -3.971</td> <td> 0.000</td> <td>   -4.129</td> <td>   -1.396</td>
</tr>
<tr>
  <th>max_age</th>          <td>    0.0501</td> <td>    0.051</td> <td>    0.981</td> <td> 0.327</td> <td>   -0.050</td> <td>    0.150</td>
</tr>
<tr>
  <th>difficulty_level</th> <td>    3.3626</td> <td>    2.064</td> <td>    1.629</td> <td> 0.104</td> <td>   -0.691</td> <td>    7.416</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>838.141</td> <th>  Durbin-Watson:     </th>  <td>   1.976</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>331324.405</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 8.064</td>  <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>121.281</td> <th>  Cond. No.          </th>  <td>2.85e+03</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.85e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



A condition number of 10-30 indicates multicollinearity, and a condition number above 30 indicates strong multicollinearity. This print-out shows a condition number of `2.77e+03`, i.e. 2770, which is well above 30.

In a predictive context (we are currently trying to build a model to assign prices to future LEGO sets, not a model primarily intended for understanding the relationship between prices and attributes of past LEGO sets), we do not *always* need to be worried when we identify strong multicollinearity. Sometimes there are features that are highly collinear but they also are individually communicating useful information to the model. In this case, however, it seems like strong multicollinearity might be what is causing our second model to have worse performance than the first model.

## Selecting Features Based on p-values

Given that we suspect our model's issues are related to multicollinearity, let's try to narrow down those features. In this case, let's use the p-values assigned to the coefficients of the model.

Looking at the model summary above, ***which features are statistically significant, with p-values above 0.05***? (P-values are labeled **P>|t|** in a StatsModels summary.)


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
const (the intercept), piece_count, and max_age
"""
```




    '\nconst (the intercept), piece_count, and max_age\n'



**Important note:** There are many limitations to using coefficient p-values to select features. See [this StackExchange answer](https://stats.stackexchange.com/a/291239) with examples in R for more details. However for now you can proceed using just the p-values technique until the more-advanced techniques have been covered.

In the cell below, create a list `significant_features` that contains the names of the columns whose features have statistically significant coefficient p-values. You should not include `"const"` in that list because `LinearRegression` from scikit-learn automatically adds a constant term and there is no column of `X_train` called `"const"`.

(You do not need to extract this information programmatically, just write them out like `"column_name_1", "column_name_2"` etc.)


```python
# Replace None with appropriate code
significant_features = None
```


```python
# __SOLUTION__
significant_features = ["piece_count", "max_age"]
```

Now let's build a model using those significant features only:


```python
# Run this cell without changes
third_model = LinearRegression()
X_train_third_model = X_train[significant_features]

third_model_scores = cross_validate(
    estimator=third_model,
    X=X_train_third_model,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Current Model")
print("Train score:     ", third_model_scores["train_score"].mean())
print("Validation score:", third_model_scores["test_score"].mean())
print()
print("Second Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```


```python
# __SOLUTION__
third_model = LinearRegression()
X_train_third_model = X_train[significant_features]

third_model_scores = cross_validate(
    estimator=third_model,
    X=X_train_third_model,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Current Model")
print("Train score:     ", third_model_scores["train_score"].mean())
print("Validation score:", third_model_scores["test_score"].mean())
print()
print("Second Model")
print("Train score:     ", second_model_scores["train_score"].mean())
print("Validation score:", second_model_scores["test_score"].mean())
print()
print("Baseline Model")
print("Train score:     ", baseline_scores["train_score"].mean())
print("Validation score:", baseline_scores["test_score"].mean())
```

    Current Model
    Train score:      0.7788368375081581
    Validation score: 0.7809488632030984
    
    Second Model
    Train score:      0.7884552982196166
    Validation score: 0.7558203636660551
    
    Baseline Model
    Train score:      0.7785726407224942
    Validation score: 0.7793473618106956


Interpret the results below. What happened when we removed the features with high p-values?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
Removing those features led to the best model so far,
although the scores are very similar to the baseline
"""
```




    '\nRemoving those features led to the best model so far,\nalthough the scores are very similar to the baseline\n'



Interesting. So, this algorithm is saying that our baseline model, with `piece_count` as the only feature, is the best one it could find.

However, note that this is based on the "importances" of the features, which means the coefficients in the context of a linear regression. It is possible that we can still get a better model by including multiple features, if we try removing columns using a different strategy.

## A Brute Force Approach

Given that we have only four columns and only a few hundred rows, one other option we have is something more computationally expensive: write custom code that goes over multiple different permutations of the columns, to see if we can find something better than the p-values approach.

The code below assumes that we want to keep the `piece_count` column, then attempts a linear regression with all possible combinations of 1-2 additional features. Don't worry too much if you don't understand everything that is happening here — an approach like this should be a last resort and you may not ever need to use it!


```python
# Run this cell without changes

from itertools import combinations

features = ["piece_count", "min_age", "max_age", "difficulty_level"]

# Make a dataframe to hold the results (not strictly necessary
# but it makes the output easier to read)
results_df = pd.DataFrame(columns=features)

# Selecting just piece_count
results_df = results_df.append({
    "train_score": baseline_scores["train_score"].mean(),
    "val_score": baseline_scores["test_score"].mean()
}, ignore_index=True)

# Selecting 1 additional feature
for feature in features[1:]:
    scores = cross_validate(
        estimator=second_model,
        X=X_train[["piece_count", feature]],
        y=y_train,
        return_train_score=True,
        cv=splitter
    )
    # Note: this technique of appending to a df is quite inefficient
    # Here it works because it's only happening 6 times, but avoid
    # doing this for a whole dataset
    results_df = results_df.append({
        feature: "Yes",
        "train_score": scores["train_score"].mean(),
        "val_score": scores["test_score"].mean()
    }, ignore_index=True)
    
# Selecting 2 additional features
for (feature1, feature2) in list(combinations(features[1:], 2)):
    scores = cross_validate(
        estimator=second_model,
        X=X_train[["piece_count", feature1, feature2]],
        y=y_train,
        return_train_score=True,
        cv=splitter
    )
    results_df = results_df.append({
        feature1: "Yes",
        feature2: "Yes",
        "train_score": scores["train_score"].mean(),
        "val_score": scores["test_score"].mean()
    }, ignore_index=True)
    
# Including all 3 additional features
results_df = results_df.append({
    "min_age": "Yes", "max_age": "Yes", "difficulty_level": "Yes",
    "train_score": second_model_scores["train_score"].mean(),
    "val_score": second_model_scores["test_score"].mean()
}, ignore_index=True)

# Fill in remaining values where appropriate
results_df["piece_count"] = "Yes"
results_df.fillna("No", inplace=True)

results_df
```


```python
# __SOLUTION__

from itertools import combinations

features = ["piece_count", "min_age", "max_age", "difficulty_level"]

# Make a dataframe to hold the results (not strictly necessary
# but it makes the output easier to read)
results_df = pd.DataFrame(columns=features)

# Selecting just piece_count
results_df = results_df.append({
    "train_score": baseline_scores["train_score"].mean(),
    "val_score": baseline_scores["test_score"].mean()
}, ignore_index=True)

# Selecting 1 additional feature
for feature in features[1:]:
    scores = cross_validate(
        estimator=second_model,
        X=X_train[["piece_count", feature]],
        y=y_train,
        return_train_score=True,
        cv=splitter
    )
    # Note: this technique of appending to a df is quite inefficient
    # Here it works because it's only happening 6 times, but avoid
    # doing this for a whole dataset
    results_df = results_df.append({
        feature: "Yes",
        "train_score": scores["train_score"].mean(),
        "val_score": scores["test_score"].mean()
    }, ignore_index=True)
    
# Selecting 2 additional features
for (feature1, feature2) in list(combinations(features[1:], 2)):
    scores = cross_validate(
        estimator=second_model,
        X=X_train[["piece_count", feature1, feature2]],
        y=y_train,
        return_train_score=True,
        cv=splitter
    )
    results_df = results_df.append({
        feature1: "Yes",
        feature2: "Yes",
        "train_score": scores["train_score"].mean(),
        "val_score": scores["test_score"].mean()
    }, ignore_index=True)
    
# Including all 3 additional features
results_df = results_df.append({
    "min_age": "Yes", "max_age": "Yes", "difficulty_level": "Yes",
    "train_score": second_model_scores["train_score"].mean(),
    "val_score": second_model_scores["test_score"].mean()
}, ignore_index=True)

# Fill in remaining values where appropriate
results_df["piece_count"] = "Yes"
results_df.fillna("No", inplace=True)

results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece_count</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>difficulty_level</th>
      <th>train_score</th>
      <th>val_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>0.778573</td>
      <td>0.779347</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>0.786925</td>
      <td>0.763876</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.778837</td>
      <td>0.780949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.778669</td>
      <td>0.780662</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0.788011</td>
      <td>0.751768</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>0.787145</td>
      <td>0.767399</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.778920</td>
      <td>0.781578</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0.788455</td>
      <td>0.755820</td>
    </tr>
  </tbody>
</table>
</div>



Interpret the table above. It shows both training and validation scores for `piece_count` as well as all combinations of 0, 1, 2, or 3 other features.

Which features make the best model? Which make the worst? How does this align with the previous discussion of multicollinearity? And how much does feature selection seem to matter in general for this dataset + model algorithm, once we have identified the most correlated feature for the baseline?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
The best model uses piece_count, max_age, and difficulty_level
It has a validation score of 0.781578

The worst model uses piece_count, min_age, and max_age
It has a validation score of 0.751768

This makes sense if we think that min_age and max_age are
mostly providing the same information, and that the difference
is mainly noise (leading to overfitting), that the best model
would only have one of them

Overall, feature selection does not seem to matter very much
for this dataset + linear regression. So long as we use our
most correlated feature (piece_count), the validation score
doesn't change very much, regardless of which other features
are included.
"""
```




    "\nThe best model uses piece_count, max_age, and difficulty_level\nIt has a validation score of 0.781578\n\nThe worst model uses piece_count, min_age, and max_age\nIt has a validation score of 0.751768\n\nThis makes sense if we think that min_age and max_age are\nmostly providing the same information, and that the difference\nis mainly noise (leading to overfitting), that the best model\nwould only have one of them\n\nOverall, feature selection does not seem to matter very much\nfor this dataset + linear regression. So long as we use our\nmost correlated feature (piece_count), the validation score\ndoesn't change very much, regardless of which other features\nare included.\n"



# 4. Build and Evaluate a Final Predictive Model

In the cell below, create a list `best_features` which contains the names of the best model features based on the findings of the previous step:


```python
# Replace None with appropriate code
best_features = None
```


```python
# __SOLUTION__
best_features = ["piece_count", "max_age", "difficulty_level"]
```

Now, we prepare the data for modeling:


```python
# Run this cell without changes
X_train_final = X_train[best_features]
X_test_final = X_test[best_features]
```


```python
# __SOLUTION__
X_train_final = X_train[best_features]
X_test_final = X_test[best_features]
```

In the cell below, instantiate a `LinearRegression` model called `final_model`, then fit it on the training data and score it on the test data.


```python
# Replace None with appropriate code

final_model = None

# Fit the model on X_train_final and y_train
None

# Score the model on X_test_final and y_test
# (use the built-in .score method)
None
```


```python
# __SOLUTION__

final_model = LinearRegression()

# Fit the model on X_train_final and y_train
final_model.fit(X_train_final, y_train)

# Score the model on X_test_final and y_test
# (use the built-in .score method)
final_model.score(X_test_final, y_test)
```




    0.6542913715071492



## User-Friendly Metrics

The score above is an r-squared score. Let's compute the MAE as well, since this would be more applicable to a business audience.


```python
# Run this cell without changes
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, final_model.predict(X_test_final))
```


```python
# __SOLUTION__
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, final_model.predict(X_test_final))
```




    14.90432119788402



What does this value mean in the current business context?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
This means that for an average LEGO set, this algorithm will
be off by about $15. Given that most LEGO sets sell for less
than $50, we would probably want to have a human double-check
and adjust these prices rather than just allowing the algorithm
to set them
"""
```




    '\nThis means that for an average LEGO set, this algorithm will\nbe off by about $15. Given that most LEGO sets sell for less\nthan $50, we would probably want to have a human double-check\nand adjust these prices rather than just allowing the algorithm\nto set them\n'



# 5. Interpret the Final Model

Display the coefficients and intercept for the final model:


```python
# Run this cell without changes
print(pd.Series(final_model.coef_, index=X_train_final.columns, name="Coefficients"))
print()
print("Intercept:", final_model.intercept_)
```


```python
# __SOLUTION__
print(pd.Series(final_model.coef_, index=X_train_final.columns, name="Coefficients"))
print()
print("Intercept:", final_model.intercept_)
```

    piece_count         0.085633
    max_age            -0.043271
    difficulty_level    2.044057
    Name: Coefficients, dtype: float64
    
    Intercept: 9.680845111984254


Interpret these values below. What is the pricing algorithm you have developed?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
According to our model, the base price for a LEGO set (the
model intercept) is about $9.68. Then for each additional
LEGO piece in the set, the price goes up by $0.09 per piece.
For every year higher that the maximum age is, the price
goes down by about $0.04. Then finally for every increase
of 1 in the difficulty level, the price goes up by about $2.04.
"""
```




    '\nAccording to our model, the base price for a LEGO set (the\nmodel intercept) is about $9.68. Then for each additional\nLEGO piece in the set, the price goes up by $0.09 per piece.\nFor every year higher that the maximum age is, the price\ngoes down by about $0.04. Then finally for every increase\nof 1 in the difficulty level, the price goes up by about $2.04.\n'



Before assuming that these coefficients give us inferential insight into past pricing decisions, we should investigate each of the assumptions of linear regression, in order to understand how much our model violates them.

## Investigating Linearity

First, let's check whether the linearity assumption holds.


```python
# Run this cell without changes

preds = final_model.predict(X_test_final)
fig, ax = plt.subplots()

perfect_line = np.arange(y_test.min(), y_test.max())
ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
ax.scatter(y_test, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();
```


```python
# __SOLUTION__
preds = final_model.predict(X_test_final)
fig, ax = plt.subplots()

perfect_line = np.arange(y_test.min(), y_test.max())
ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
ax.scatter(y_test, preds, alpha=0.5)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.legend();
```


    
![png](index_files/index_107_0.png)
    


Are we violating the linearity assumption?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
We have some outliers that are all over the
place, but in general it looks like we have
a linear relationship (not violating this
assumption)
"""
```




    '\nWe have some outliers that are all over the\nplace, but in general it looks like we have\na linear relationship (not violating this\nassumption)\n'



### Investigating Normality

Now let's check whether the normality assumption holds for our model.


```python
# Run this code without changes
import scipy.stats as stats

residuals = (y_test - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);
```


```python
# __SOLUTION__
import scipy.stats as stats
residuals = (y_test - preds)
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);
```


    
![png](index_files/index_113_0.png)
    


Are we violating the normality assumption?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
Our outliers are again causing problems. This
is bad enough that we can probably say that we
are violating the normality assumption
"""
```




    '\nOur outliers are again causing problems. This\nis bad enough that we can probably say that we\nare violating the normality assumption\n'



### Investigating Multicollinearity (Independence Assumption)

Another way to measure multicollinearity is with [variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) ([StatsModels documentation here](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html#statsmodels.stats.outliers_influence.variance_inflation_factor)). A "rule of thumb" for VIF is that 5 is too high (i.e. strong multicollinearity).

Run the code below to find the VIF for each feature.


```python
# Run this cell without changes
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train_final.values, i) for i in range(X_train_final.shape[1])]
pd.Series(vif, index=X_train_final.columns, name="Variance Inflation Factor")
```


```python
# __SOLUTION__ 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train_final.values, i) for i in range(X_train_final.shape[1])]
pd.Series(vif, index=X_train_final.columns, name="Variance Inflation Factor")
```




    piece_count         1.923641
    max_age             1.689177
    difficulty_level    1.965106
    Name: Variance Inflation Factor, dtype: float64



Do we have too high of multicollinearity?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
We are below 5 for all features in the final model,
so we don't have too high of multicollinearity
"""
```




    "\nWe are below 5 for all features in the final model,\nso we don't have too high of multicollinearity\n"



### Investigating Homoscedasticity

Now let's check whether the model's errors are indeed homoscedastic or if they violate this principle and display heteroscedasticity.


```python
# Run this cell without changes
fig, ax = plt.subplots()

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(X_test))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");
```


```python
# __SOLUTION__ 
fig, ax = plt.subplots()

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(X_test))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual - Predicted Value");
```


    
![png](index_files/index_125_0.png)
    


Are we violating the homoscedasticity assumption?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
This is not the worst "funnel" shape, although
the residuals do seem to differ some based on
the predicted price. We are probably violating
a strict definition of homoscedasticity.
"""
```




    '\nThis is not the worst "funnel" shape, although\nthe residuals do seem to differ some based on\nthe predicted price. We are probably violating\na strict definition of homoscedasticity.\n'



### Linear Regression Assumptions Conclusion

Given your answers above, how should we interpret our model's coefficients? Do we have a model that can be used for inferential as well as predictive purposes? What might your next steps be?


```python
# Replace None with appropriate text
"""
None
"""
```


```python
# __SOLUTION__
"""
Our confidence in these coefficients should not be too high, since
we are violating or close to violating more than one of the
assumptions of linear regression. This really only should be used
for predictive purposes.

A good next step here would be to start trying to figure out why
our outliers behave the way they do. Maybe there is some information
we could extract from the text features that are currently not part
of the model
"""
```




    '\nOur confidence in these coefficients should not be too high, since\nwe are violating or close to violating more than one of the\nassumptions of linear regression. This really only should be used\nfor predictive purposes.\n\nA good next step here would be to start trying to figure out why\nour outliers behave the way they do. Maybe there is some information\nwe could extract from the text features that are currently not part\nof the model\n'



# 6. Create Presentation Notebook

Now that you've completed your project, let's put it into an easily presentable format so you can add it to your portfolio. To do this, we recommend completing the following steps outside of this notebook.

1. Create a new GitHub repository for your project.
2. Save a copy of this notebook and the `data` subfolder into your local repository.
3. Edit the text and images in the notebook to present your project and help someone else understand it.
4. Run your notebook from start to finish, then save it.
5. Create a README.md file in your repository with a brief summary of your project.
6. Push your updated repository to GitHub to share with your instructor and employers!

# Level Up: Project Enhancements

After completing the project, you could consider the following enhancements if you have time:

* Engineer new features to improve the predictive power of your model
* Identify and remove outliers, then redo the analysis
* Identify models with high or low value for LEGO buyers, using the differences between actual and predicted prices
* Conduct statistical tests using the numeric features in the dataset to make inferences about the population of LEGO sets

## Summary

Well done! As you can see, regression can be a challenging task that requires you to make decisions along the way, try alternative approaches, and make ongoing refinements.
