# Module 3: TRD

We want to build a product that relies on a predictive model that allow us to target a set of users that are highly likely to be interested in an item of our choice that we want to promote to send them a push notification. More specifically, we will be developing a machine learning model that, given a user and a product, predicts if the user would purchase it if they were buying with us at that point in time. 

This way, we will be sending push notifications to our users is an effective manner to boost sales and offer discounts to incentivate user engagement with our targeted products without been intrusive and sending too many notifications, since this can generate user disatisfaction and generate churn.

It is worth mentioning that the target impact is to increase our monthly sales by 2% and a boost of 25% over the selected items.

For this product, we have two requirements:
* We are only interested in users that purchase the item along with at least other 4 (minimum 5 items basket)
* The system should allow sales operators to select an item from a dropdown or search bar, get the segment of users to target and trigger a customizable push notification.


## 0. Importing necessary libraries


```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler


```

## 1. Data processing

In this dataset, every row representes an (order, product) pair where outcome indicates whether the product was bought or not on that specific order and every other feature has been computed only looking at information prior to that order in order to avoid information leakage.

It is worth remembering that the column outcome indicates, for each order, if a certain product has been bought (for the whole catalogue of products). Therefore, there are lots of 0 outcome instances, and the dataset is imbalanced.


```python
file_path = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/data/groceries/box_builder_dataset/feature_frame.csv")

# Temporarily adjust the max columns displayed
pd.set_option('display.max_columns', None)

data = pd.read_csv(file_path)
data.head()
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>vendor</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (2880549, 27)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
information_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
target_col = 'outcome'
feature_cols = [col for col in data.columns if col not in information_cols + [target_col]]

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols = [col for col in feature_cols if col not in categorical_cols + binary_cols]
```


```python
data['variant_id'] = data['variant_id'].astype('str')
data['order_id'] = data['order_id'].astype('str')
data['user_id'] = data['user_id'].astype('str')

data['created_at'] = data['created_at'].astype('datetime64[us]')
data['order_date'] = data['order_date'].astype('datetime64[us]')

data['outcome'] = data['outcome'].astype(int)

for col in binary_cols:
    data[col] = data[col].astype(int)

# Iterate over each column and check if it starts with 'count_'
for col in data.columns:
    if col.startswith('count_'):
        # Convert the column to integer type
        data[col] = data[col].astype(int)

data['people_ex_baby'] = data['people_ex_baby'].astype('int64')

data['days_since_purchase_variant_id'] = data['days_since_purchase_variant_id'].astype('int64')
data['days_since_purchase_product_type'] = data['days_since_purchase_product_type'].astype('int64')
```

It is worth mentioning that, since we have being working on this dataset over the latest several weeks, this project skips the Exploratory Data Analysis (EDA) phase as it is already available in previous reports.

## Milestone 1: Exploration Phase
### 1. Building the dataset

#### 1.1 Filtering the data to orders with at least 5 items

Firstly, we filter the data to only those orders with at least 5 items to build a dataset to work with, since they are our target. 


```python
# Group data by order_id and sum the outcome (A product is in the order if its outcome is 1)
order_size = data.groupby('order_id').outcome.sum()

# Identify orders with size at least 5 and extract the order_id of these big orders
big_orders = order_size[order_size >= 5].index 

# Filter data to only include rows where order_id is in the list of big_orders
filtered_data = data.loc[lambda x: x.order_id.isin(big_orders)]
```


```python
filtered_data.shape
```




    (2163953, 27)



#### 1.2 Splitting the data


```python
# Get how many orders are performed each day
daily_orders = filtered_data.groupby('order_date').order_id.nunique()
daily_orders.head()
```




    order_date
    2020-10-05     3
    2020-10-06     7
    2020-10-07     6
    2020-10-08    12
    2020-10-09     4
    Name: order_id, dtype: int64




```python
plt.plot(daily_orders, label="Daily orders")
plt.title("Daily Orders")
```




    Text(0.5, 1.0, 'Daily Orders')




    
![png](Module3_Milestone1_files/Module3_Milestone1_21_1.png)
    


**Important**: It's evident that there is a significant time-based aspect, signaling changes in patterns and business evolution over time. Consequently, it is prudent to implement a temporal split. This approach not only captures the dynamic nature of the data but also ensures that the same order does not appear in both the training and testing sets, thereby preventing information leakage.


```python
cum_sum_daily_orders = daily_orders.cumsum() / daily_orders.sum()

train_val_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.7].idxmax()
val_test_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.9].idxmax()

print("Train set from: ", cum_sum_daily_orders.index.min())
print("Train set until: ", train_val_cut)
print("Validation set from: ", val_test_cut)
print("Test set until: ", cum_sum_daily_orders.index.max())
```

    Train set from:  2020-10-05 00:00:00
    Train set until:  2021-02-04 00:00:00
    Validation set from:  2021-02-22 00:00:00
    Test set until:  2021-03-03 00:00:00


It is important to consider what is the minimum business cicle, that is, the minimum temporal window that allows us to capture the complete dynamics of the data we want to model. Even though the validation and test splits don't comprise many days, we consider these should be sufficient for our case.


```python
train_data = filtered_data[filtered_data.order_date <= train_val_cut]
val_data = filtered_data[(filtered_data.order_date > train_val_cut) & (filtered_data.order_date <= val_test_cut) ]
test_data = filtered_data[filtered_data.order_date > val_test_cut]
```

Divide into features and target:


```python
def feature_target_split(df, target):
    X = df.drop(columns=[target])  # Drop the target column to create the features DataFrame
    Y = df[target]   # Target variable we want to predict
    return X, Y

X_train, Y_train = feature_target_split(train_data, target_col)
X_val, Y_val = feature_target_split(val_data, target_col)
X_test, Y_test = feature_target_split(test_data, target_col)
```

### 2. Defining a Baseline model

In order to be able to decide if a ML model generates any value, we need to compare it against baselines that do not require training. In this case, we will use the global popularity feature as baseline, that is, the product will be predicted to be bought the more popular it is.


```python
def plot_curves(y_true, y_pred, curve_type="both", dataset_type="train"):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{dataset_type.capitalize()} Dataset - Performance Curves")

    if curve_type in ["precision-recall", "both"]:
        # Compute the precision-recall curve and its AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        axes[0].step(recall, precision)
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title(f'Precision-Recall Curve (AUC={pr_auc:.2f})')

    if curve_type in ["roc", "both"]:
        # Compute the ROC curve and its AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        axes[1].plot(fpr, tpr)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve (AUC={roc_auc:.2f})')

    plt.tight_layout()
    plt.show()

```


```python
y_pred = val_data['global_popularity']
y_true = val_data['outcome']


plot_curves(y_true, y_pred, curve_type="both", dataset_type="train")

```


    
![png](Module3_Milestone1_files/Module3_Milestone1_30_0.png)
    


### 3. Training an initial model with only numerical variables
I will be trainning a logistic regression model as the target variable is binary.
Additionally, I start by training some initial models that do not need preprocessing, that is, with only numerical (and binary) variables. 


```python
train_cols = numerical_cols + binary_cols
```

#### 3.1 Ridge Regression

**Important:** Before training the Ridge Regression model, it is crucial to standardize the data, ensuring each column has a mean of 0 and a standard deviation of 1. This standardization makes all features comparable, giving equal importance to each one in the model. This practice prevents variables with larger scales from disproportionately influencing the model's predictions due to their scale rather than their relevance. Additionally, standardizing the data helps in achieving faster and more stable convergence of the regression algorithm.


```python
# First, standardize the data
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train[train_cols])


# Apply the same transformation to the validation data
X_val_scaled = scaler.transform(X_val[train_cols])

```


```python
def plot_curves_with_hyperparameter(ax, y_true, y_pred, label, curve_type="both"):
    if curve_type in ["precision-recall", "both"]:
        # Compute the precision-recall curve and its AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        ax[0].step(recall, precision, label=f'{label} (AUC={pr_auc:.2f})')
        ax[0].set_xlabel('Recall')
        ax[0].set_ylabel('Precision')
        ax[0].set_title('Precision-Recall Curve')

    if curve_type in ["roc", "both"]:
        # Compute the ROC curve and its AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        ax[1].plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.2f})')
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('ROC Curve')
```


```python
# Train the Logistic Regression model

# Prepare the plot environments for both training and validation
fig_train, axes_train = plt.subplots(1, 2, figsize=(14, 6))
fig_train.suptitle('Training Data - Performance Curves Across Different C Values')
fig_val, axes_val = plt.subplots(1, 2, figsize=(14, 6))
fig_val.suptitle('Validation Data - Performance Curves Across Different C Values')


# Define hyperparameter for the level of regularization
cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]


for c in cs:
    if c == None:
        lr = LogisticRegression(penalty = None)
    else:
        lr = LogisticRegression(penalty='l2', C=c)
    
    # Fit the model on scaled training data
    lr.fit(X_train_scaled, Y_train)

    # Predict probabilities for the positive class on the training set
    train_proba = lr.predict_proba(X_train_scaled)[:, 1]
    plot_curves_with_hyperparameter(axes_train, Y_train, train_proba, f'Train C={c}', curve_type="both")


    # Predict probabilities for the positive class on the validation set
    val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    plot_curves_with_hyperparameter(axes_val, Y_val, val_proba, f'Val C={c}', curve_type="both")

# Add legends and show plots
axes_train[0].legend(loc='upper right')
axes_train[1].legend(loc='lower right')
axes_val[0].legend(loc='upper right')
axes_val[1].legend(loc='lower right')

plt.tight_layout()
fig_train.show()
fig_val.show()
```

    /tmp/ipykernel_82778/2186894484.py:39: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_train.show()
    /tmp/ipykernel_82778/2186894484.py:40: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_val.show()



    
![png](Module3_Milestone1_files/Module3_Milestone1_36_1.png)
    



    
![png](Module3_Milestone1_files/Module3_Milestone1_36_2.png)
    


Firstly, our analysis indicates that regularization does not enhance the performance of our model. Both the ROC and precision-Recall curves, along with their respective areas under the curve (AUC), demonstrate consistent results across various regularization strengths. This consistency suggests that the existing complexity of the model is well-suited to our task, as the introduction of regularization does not yield measurable improvements. The lack of performance variance with different levels of regularization implies that the model's complexity is already optimal, negating the need for further regularization adjustments.

Furthermore, it is worth mentioning that our dataset is significantly imbalanced, as previously mentioned, with a predominance of negative examples. This imbalance skews the False Positive Rate (FPR), which remains deceptively low even when the absolute number of false positives is substantial. In contrast, the Precision-Recall curves are more revealing, demonstrating notable improvements over the baseline.

Finally, comparing the model’s performance on training and validation datasets reveals similar error rates, suggesting effective generalization to new data. This consistency confirms that the model is not overfitting; it is not merely memorizing the training data or modelling its noise, but is learning to generalize from it.

#### 3.2 Lasso Regression



```python
# Train the Logistic Regression model

# Prepare the plot environments for both training and validation
fig_train, axes_train = plt.subplots(1, 2, figsize=(14, 6))
fig_train.suptitle('Training Data - Performance Curves Across Different C Values')
fig_val, axes_val = plt.subplots(1, 2, figsize=(14, 6))
fig_val.suptitle('Validation Data - Performance Curves Across Different C Values')


# Define hyperparameter for the level of regularization
cs = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4, None]


for c in cs:
    if c == None:
        lr = LogisticRegression(penalty = None)
    else:
        lr = LogisticRegression(penalty='l1', C=c, solver="saga")
    
    # Fit the model on scaled training data
    lr.fit(X_train_scaled, Y_train)

    # Predict probabilities for the positive class on the training set
    train_proba = lr.predict_proba(X_train_scaled)[:, 1]
    plot_curves_with_hyperparameter(axes_train, Y_train, train_proba, f'Train C={c}', curve_type="both")


    # Predict probabilities for the positive class on the validation set
    val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    plot_curves_with_hyperparameter(axes_val, Y_val, val_proba, f'Val C={c}', curve_type="both")

# Add legends and show plots
axes_train[0].legend(loc='upper right')
axes_train[1].legend(loc='lower right')
axes_val[0].legend(loc='upper right')
axes_val[1].legend(loc='lower right')

plt.tight_layout()
fig_train.show()
fig_val.show()
```

    /tmp/ipykernel_82778/1816860315.py:39: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_train.show()
    /tmp/ipykernel_82778/1816860315.py:40: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_val.show()



    
![png](Module3_Milestone1_files/Module3_Milestone1_39_1.png)
    



    
![png](Module3_Milestone1_files/Module3_Milestone1_39_2.png)
    


This time, with Lasso regression, we can observe that for the Precision-Recall plot indicates that the model works best for $c=10^{-8}$ and $c=10^{-6}$ the model performs the best, as it contais a higher AUC. Nevertheless, when looking at the ROC curve, we can observe that these values provide the lowest AUC. Therefore, the level of regularization does not seem to make a great impact. 

Additionally, we can observe that the performance is similar to Ridge.

Disclaimer: it is worth mentioning that the Precision-Recall curve for $c=10^{-6}$ does not seem correct, as it does not match with its corresponding AUC.

### 3.3. Coefficients weights


```python
c_l2 = 10e-6
lr = LogisticRegression(penalty='l2', C=c_l2)
lr.fit(X_train_scaled, Y_train)

lr_coefs_l2 = pd.DataFrame({"features": train_cols, 
                            "importance": np.abs(lr.coef_[0]), 
                            "regularisation" : ["l2"]*len(train_cols)})

lr_coefs_l2 = lr_coefs_l2.sort_values('importance', ascending=True)

c_l1 = 10e-4
lr = LogisticRegression(penalty='l1', C=c_l1,  solver="saga")
lr.fit(X_train_scaled, Y_train)

lr_coefs_l1 = pd.DataFrame({"features": train_cols, 
                            "importance": np.abs(lr.coef_[0]), 
                            "regularisation" : ["l1"]*len(train_cols)})

lr_coefs_l1 = lr_coefs_l1.sort_values('importance', ascending=True)
```


```python
lr_coefs = pd.concat([lr_coefs_l1, lr_coefs_l2])
lr_coefs["features"] = pd.Categorical(lr_coefs["features"])

# Sort again
lr_coefs = lr_coefs.sort_values('importance', ascending=True)
order_cols = lr_coefs_l2.sort_values('importance', ascending=False)["features"]

sns.barplot(lr_coefs, x="importance", y="features", hue="regularisation", order=order_cols)
```




    <Axes: xlabel='importance', ylabel='features'>




    
![png](Module3_Milestone1_files/Module3_Milestone1_43_1.png)
    


we can observe that, as expected, L1 and L2 have different behaviour regarding how many features are assigned $\beta = 0$. Lasso (L1) has 6 variables, as it tends to give 0 weights, whereas Ridge has more variables with lower weights. Since we can observe that several features are not relevant, as they are assigned 0 (or very low) weights, we will train a model only with the significant ones.

### 4. Training models with only a subset of features


```python
reduced_cols = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']
X_train_reduced_scaled = scaler.fit_transform(X_train[reduced_cols])
X_val_reduced_scaled = scaler.fit_transform(X_val[reduced_cols])
```


```python
# Train the Logistic Regression model

# Prepare the plot environments for both training and validation
fig_train, axes_train = plt.subplots(1, 2, figsize=(14, 6))
fig_train.suptitle('Training Data - Performance Curves Across Different C Values')
fig_val, axes_val = plt.subplots(1, 2, figsize=(14, 6))
fig_val.suptitle('Validation Data - Performance Curves Across Different C Values')


## L2
# Define hyperparameter for the level of regularization
c_l2 = 10e-6

lr2 = LogisticRegression(penalty='l2', C=c_l2)

# Fit the model on scaled training data
lr2.fit(X_train_reduced_scaled, Y_train)

# Predict probabilities for the positive class on the training set
train_proba2 = lr2.predict_proba(X_train_reduced_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_train, Y_train, train_proba2, 'Train L2', curve_type="both")

# Predict probabilities for the positive class on the validation set
val_proba2 = lr2.predict_proba(X_val_reduced_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_val, Y_val, val_proba2, 'Val L2', curve_type="both")

## L1
c_l1 = 10e-4
lr1 = LogisticRegression(penalty='l1', C=c_l1, solver="saga")
    
# Fit the model on scaled training data
lr1.fit(X_train_reduced_scaled, Y_train)

# Predict probabilities for the positive class on the training set
train_proba1 = lr1.predict_proba(X_train_reduced_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_train, Y_train, train_proba1, 'Train L1', curve_type="both")

# Predict probabilities for the positive class on the validation set
val_proba1 = lr1.predict_proba(X_val_reduced_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_val, Y_val, val_proba1, 'Val L1', curve_type="both")


# Add legends and show plots
axes_train[0].legend(loc='upper right')
axes_train[1].legend(loc='lower right')
axes_val[0].legend(loc='upper right')
axes_val[1].legend(loc='lower right')

plt.tight_layout()
fig_train.show()
fig_val.show()
```

    /tmp/ipykernel_82778/2610710267.py:50: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_train.show()
    /tmp/ipykernel_82778/2610710267.py:51: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_val.show()



    
![png](Module3_Milestone1_files/Module3_Milestone1_47_1.png)
    



    
![png](Module3_Milestone1_files/Module3_Milestone1_47_2.png)
    


We can see that the model's performance remains consistent with previous outcomes, despite using only a limited set of variables for both models.

### 5. Training a model with categorical variables

#### Addressing categorical variables
Logistic regression models in libraries like Scikit-learn expect all input features to be numeric. Therefore, categorical variables need to be converted into a numerical format before they can be used for model training. It is worth remembering that we do not consider variant_id, order_id, and user_id categorical variables, but informative ones, and these will not be used to train the model.

First, deal with the two datetime columns. For created_at, I split the information of these into four columns: for year, month, day and hour. On the other hand, since order_date does not contain any additional information, I remove it.


```python
def transform_datetime(df, date_col):
    """
    Transforms a datetime column into the following extracted components: year, month, day, and hour.
    
    Input parameters:
    df (pd.DataFrame): The DataFrame containing the datetime column.
    date_col (datetome): The name of the datetime column to transform.
    
    Returns:
    pd.DataFrame: The modified DataFrame with new columns for date components.
    """

    # Check if the column exists in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"The column {date_col} does not exist in the DataFrame.")
    
    # Extracting components
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_hour'] = df[date_col].dt.hour
    
    # Remove the original datetime column
    df = df.drop(columns=[date_col])
    
        
    return df
```


```python
extended_cols = reduced_cols + categorical_cols + ['created_at']
X_train_cat = X_train[extended_cols]
X_val_cat = X_val[extended_cols]

# Transform created_at column
X_train_cat = transform_datetime(X_train_cat, 'created_at')
X_val_cat = transform_datetime(X_val_cat, 'created_at')
   
print(X_train_cat.shape)
X_train_cat.head()
```

    /tmp/ipykernel_82778/969680544.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_year'] = df[date_col].dt.year
    /tmp/ipykernel_82778/969680544.py:19: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_month'] = df[date_col].dt.month
    /tmp/ipykernel_82778/969680544.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_day'] = df[date_col].dt.day
    /tmp/ipykernel_82778/969680544.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_hour'] = df[date_col].dt.hour


    (1426520, 10)


    /tmp/ipykernel_82778/969680544.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_year'] = df[date_col].dt.year
    /tmp/ipykernel_82778/969680544.py:19: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_month'] = df[date_col].dt.month
    /tmp/ipykernel_82778/969680544.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_day'] = df[date_col].dt.day
    /tmp/ipykernel_82778/969680544.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[f'{date_col}_hour'] = df[date_col].dt.hour





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
      <th>ordered_before</th>
      <th>global_popularity</th>
      <th>abandoned_before</th>
      <th>normalised_price</th>
      <th>product_type</th>
      <th>vendor</th>
      <th>created_at_year</th>
      <th>created_at_month</th>
      <th>created_at_day</th>
      <th>created_at_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>ricepastapulses</td>
      <td>clearspring</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>ricepastapulses</td>
      <td>clearspring</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>ricepastapulses</td>
      <td>clearspring</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>ricepastapulses</td>
      <td>clearspring</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>ricepastapulses</td>
      <td>clearspring</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



Now, for the categorical variables (product_type and vendor), which are strings, my first idea was to use Dummy variables for all, which are similar to one-hot encoding but avoids one redundant category by dropping the first category.

But first, I check how many categories each column contains to address if it is feasable to do dummy variables or if too many columns would be created.


```python
for col in categorical_cols:
    print("Number of cateogories in: ", col, ": ", len(X_train_cat[col].unique()))  # Returns array of unique values

```

    Number of cateogories in:  product_type :  59
    Number of cateogories in:  vendor :  240


We can observe that the only column where using the Dummy variables approach may be feasible is for product_type. Furthermore, since  vendor would create too many columns with this approach (as it has many different variables), I try One Hot encoder (with colisions).


```python
"""# Initialize OneHotEncoder
OHE = OneHotEncoder(max_categories=25, sparse_output=False)  

X_train_encoded = pd.DataFrame(index=X_train_cat.index)

for col in categorical_cols:
    # Fit and transform data
    encoded_data = OHE.fit_transform(X_train_cat[[col]])
    
    # Get categories for the current column from the encoder
    categories = OHE.categories_[0]
    
    # Create column names for the encoded features
    col_names = [f"{col}_{category}" for category in categories]
    
    # Ensure the DataFrame is created with matching number of columns
    encoded_df = pd.DataFrame(encoded_data, columns=col_names, index=X_train_cat.index)
    
    # Concatenate the new DataFrame to the existing DataFrame
    X_train_encoded = pd.concat([X_train_encoded, encoded_df], axis=1)"""

```




    '# Initialize OneHotEncoder\nOHE = OneHotEncoder(max_categories=25, sparse_output=False)  \n\nX_train_encoded = pd.DataFrame(index=X_train_cat.index)\n\nfor col in categorical_cols:\n    # Fit and transform data\n    encoded_data = OHE.fit_transform(X_train_cat[[col]])\n    \n    # Get categories for the current column from the encoder\n    categories = OHE.categories_[0]\n    \n    # Create column names for the encoded features\n    col_names = [f"{col}_{category}" for category in categories]\n    \n    # Ensure the DataFrame is created with matching number of columns\n    encoded_df = pd.DataFrame(encoded_data, columns=col_names, index=X_train_cat.index)\n    \n    # Concatenate the new DataFrame to the existing DataFrame\n    X_train_encoded = pd.concat([X_train_encoded, encoded_df], axis=1)'




```python
# Label Encoding
label_encoder = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train_cat[col] = le.fit_transform(X_train_cat[col])
    X_val_cat[col] = le.fit_transform(X_val_cat[col])
    label_encoder[col] = le  # Store the label encoder for each column

# Concatenate with non-categorical features
non_categorical_cols = [col for col in X_train_cat.columns if col not in categorical_cols]
X_train_final = pd.concat([X_train_cat[non_categorical_cols], X_train_cat], axis=1)
X_val_final = pd.concat([X_val_cat[non_categorical_cols], X_val_cat], axis=1)

X_train_final.head()
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
      <th>ordered_before</th>
      <th>global_popularity</th>
      <th>abandoned_before</th>
      <th>normalised_price</th>
      <th>created_at_year</th>
      <th>created_at_month</th>
      <th>created_at_day</th>
      <th>created_at_hour</th>
      <th>ordered_before</th>
      <th>global_popularity</th>
      <th>abandoned_before</th>
      <th>normalised_price</th>
      <th>product_type</th>
      <th>vendor</th>
      <th>created_at_year</th>
      <th>created_at_month</th>
      <th>created_at_day</th>
      <th>created_at_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>46</td>
      <td>44</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>17</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>46</td>
      <td>44</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.081052</td>
      <td>46</td>
      <td>44</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>46</td>
      <td>44</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>0.038462</td>
      <td>0</td>
      <td>0.081052</td>
      <td>46</td>
      <td>44</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Train the Logistic Regression model
X_train_final_scaled = scaler.fit_transform(X_train_final[reduced_cols])
X_val_final_scaled = scaler.fit_transform(X_val_final[reduced_cols])

# Prepare the plot environments for both training and validation
fig_train, axes_train = plt.subplots(1, 2, figsize=(14, 6))
fig_train.suptitle('Training Data - Performance Curves')
fig_val, axes_val = plt.subplots(1, 2, figsize=(14, 6))
fig_val.suptitle('Validation Data - Performance Curves')

c = 1e-6
lr = LogisticRegression(penalty='l2', C=c)

# Fit the model on scaled training data
lr.fit(X_train_final_scaled, Y_train)

# Predict probabilities for the positive class on the training set
train_proba = lr.predict_proba(X_train_final_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_train, Y_train, train_proba, f'Train C={c}', curve_type="both")


# Predict probabilities for the positive class on the validation set
val_proba = lr.predict_proba(X_val_final_scaled)[:, 1]
plot_curves_with_hyperparameter(axes_val, Y_val, val_proba, f'Val C={c}', curve_type="both")

# Add legends and show plots
axes_train[0].legend(loc='upper right')
axes_train[1].legend(loc='lower right')
axes_val[0].legend(loc='upper right')
axes_val[1].legend(loc='lower right')

plt.tight_layout()
fig_train.show()
fig_val.show()
```

    /tmp/ipykernel_82778/1566344662.py:33: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_train.show()
    /tmp/ipykernel_82778/1566344662.py:34: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig_val.show()



    
![png](Module3_Milestone1_files/Module3_Milestone1_60_1.png)
    



    
![png](Module3_Milestone1_files/Module3_Milestone1_60_2.png)
    


We can observe that adding categorical variables does not make the model improve in its performance.

### 6. Evaluating the final model

The final model selected will be the logistic regression with L2 penalty and c=1e-3. Additionally, the model will only be trained with the subset of numerical column previously defined.


```python
lr = LogisticRegression(penalty = 'l2', C = 1e-6)

# Fit the model on scaled training data
lr.fit(X_train_reduced_scaled, Y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=1e-06)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=1e-06)</pre></div> </div></div></div></div>




```python
# Predict and test the model
Y_pred = lr.predict(X_val_reduced_scaled)
accuracy = accuracy_score(Y_val, Y_pred)
report = classification_report(Y_val, Y_pred)
```

    /home/adriana/.cache/pypoetry/virtualenvs/zrive-ds-3iE0R8j--py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/adriana/.cache/pypoetry/virtualenvs/zrive-ds-3iE0R8j--py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /home/adriana/.cache/pypoetry/virtualenvs/zrive-ds-3iE0R8j--py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))



```python
print("Accuracy: ", accuracy)
```

    Accuracy:  0.9864797576146309


This value in the accuracy indicates that the model correctly predicts both negatives and positives 98.9% of the time. While this may initially seem excellent, it's crucial to delve deeper, especially into how it performs on each class, due to possible imbalance in class distribution.


```python
print(report)
```

                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99    467548
               1       0.00      0.00      0.00      6408
    
        accuracy                           0.99    473956
       macro avg       0.49      0.50      0.50    473956
    weighted avg       0.97      0.99      0.98    473956
    



```python
# Generate the confusion matrix
conf_matrix = confusion_matrix(Y_val, Y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)
```

    Confusion Matrix:
    [[467548      0]
     [  6408      0]]


It is worth mentioning the confusion matrix, the true negatives are located in the top-left, the false positives in the top-right, the false negatives in the bottom-left, and, finally, in the bottom right, the true positives. 

The first thing we can observe from this report is that there is a clear imbalance between classes: class 0 (negative) represents the majority class, with 467548, whereas class 1 (positive) contains only 6408 instances.

On one hand, for the negative class (0), from the high precision, we can observe that nearly all instances predicted as class 0 are indeed class 0, which suggests very few false positives for class 0. Furthermore, the recall is 100%, indicating that almost all 0 instances are correctly predicted by the model. Finally, a high F1 score for class 0 indicates excellent precision and recall balance. In conclusion, the model is very effective at identifying class 0 instances. All this can also be observed in the first row of the confusion matrix, as the number of true  negatives is 467455 and the number of false positives is only 93 (the model incorrectly predicted 93 cases as positive when they were actually negative).

On the other hand, for the positive class (1), the results are not as good. Its precision of 72% indicates that, of all instances predicted as 1, only 72% actually belonged to class 1. Additionally, the model's recall for this class is extremely low: only 4% of the actual cases with outcome 1  are correctly identified, indicating the model struggles significantly to detect class 1 instances. Finally, as we would have expected, the F1 score is very low, reflecting how poor performance is in terms of both precision and recall. Once again, this can also be seen in the confusion matrix: it has a really low number of true positives (245), and a really high number of false positives (6163).

In conclusion, we can observe that the model is very effective at identifying class 0 intances, but for cases where the outcome is 1 it performs extremely poorly,  indicating the model is biased towards the majority class, as it has been trained on an imbalanced dataset. 
