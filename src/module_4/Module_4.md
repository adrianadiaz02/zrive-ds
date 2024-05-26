# Module 4: TDD

We want to build a product that relies on a predictive model that allow us to target a set of users that are highly likely to be interested in an item of our choice that we want to promote to send them a push notification. More specifically, we will be developing a machine learning model that, given a user and a product, predicts if the user would purchase it if they were buying with us at that point in time. 

This way, we will be sending push notifications to our users is an effective manner to boost sales and offer discounts to incentivate user engagement with our targeted products without been intrusive and sending too many notifications, since this can generate user disatisfaction and generate churn.

It is worth mentioning that the target impact is to increase our monthly sales by 2% and a boost of 25% over the selected items.

For this product, we have two requirements:
* We are only interested in users that purchase the item along with at least other 4 (minimum 5 items basket)
* The system should allow sales operators to select an item from a dropdown or search bar, get the segment of users to target and trigger a customizable push notification.


## 0. Importing necessary libraries


```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, log_loss, precision_recall_curve, roc_auc_score, roc_curve

from sklearn.pipeline import make_pipeline
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

It is worth mentioning that, since we have being working on this dataset over the latest several weeks, this project skips the Exploratory Data Analysis (EDA) phase as it is already available in previous reports. If you want to have a look at the EDA performed, please check these two files:
- https://github.com/adrianadiaz02/zrive-ds/blob/main/src/module_2/Module_2_EDA_Ex1.md (Understanding the problem space)
- https://github.com/adrianadiaz02/zrive-ds/blob/main/src/module_2/Module_2_EDA_Ex2.md (EDA)

## Milestone 1: Exploration Phase
### 1. Building the dataset

#### 1.1 Filtering the data to orders with at least 5 items

Firstly, we filter the data to only those orders with at least 5 items to build a dataset to work with, since they are our target. 


```python
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
fig, ax = plt.subplots()
ax.plot(daily_orders, label="Daily orders")
ax.set_title("Daily Orders")
ax.legend()
plt.show()
```


    
![png](Module_4_files/Module_4_19_0.png)
    


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
    X = df.drop(columns=[target]) 
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
    """ Plot the Precision-Recall and ROC curves, together with their AUC.
    Curve_type can be "precision-recall", "roc", or "both", and determine what  are the plotted curves.
    Dataset_type is used to plot in the title wether the the predictions are done in training, test or validation."""
    
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


    
![png](Module_4_files/Module_4_28_0.png)
    


We can observe the AUC of the Precision-Recall plot is of 0.07 and for the ROC curve it is 0.79.

### 3. Linear Models
In the previous module, we explored different linear models (logistic regression) with different parametrisations, regularisations (Ridge and Lasso), and set of features. If interested, please consult the corresponding notebook:
- https://github.com/adrianadiaz02/zrive-ds/blob/main/src/module_3/Module3_Milestone1.md 

Nevertheless, it is worth mentioning that the final model chosen was a Logistic Regression with L2 penalty, an hyperparameter value for C of 1e-4, and only the subset of features from the numerical columns ('ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price):


```python
# Define the columns to be used
reduced_cols = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']

# Define hyperparameter for the level of regularization
c = 1e-3

# Create a pipeline with StandardScaler and LogisticRegression
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', C=1e-4)
)

# Fit the model on the training data
lr.fit(X_train[reduced_cols], Y_train)

# Predict scores for the positive class on the training set
train_proba2 = lr.predict_proba(X_train[reduced_cols])[:, 1]
plot_curves(Y_train, train_proba2, curve_type="both", dataset_type="train")

# Predict score for the positive class on the validation set
val_proba2 = lr.predict_proba(X_val[reduced_cols])[:, 1]
plot_curves(Y_val, val_proba2, curve_type="both", dataset_type="val")

plt.show()

```


    
![png](Module_4_files/Module_4_31_0.png)
    



    
![png](Module_4_files/Module_4_31_1.png)
    


As we can observe, this model provided an AUC of the Precision-Recall curve of 0.16 and 0.14 for the training and validation sets respectively. On the other hand, the AUC of the ROC curve had a value of 0.81 for both subsets.

### 4. Non-linear Models
We will now try to improve our model by increasing its complexity.

#### 4.1 Random Forest
In a Random Forest, each tree receives a subset of the data, and each split only looks at a random subset of the features. Additionaly, the trees are constructed until the end and the decision is made by a voting process. Furthermore, it is worth mentioning that is hass three hyperparameters: the number of trees, the variables at each split, and the sample size at each tree. Nevertheless, we will only define the number of trees, and set the other hyperparameters with their default value.

It is also worth mentioning that now we do not need to scale the data, as there is no need to add regularisation.


```python
def evaluate_configuration(model, X_train, Y_train, X_val, Y_val):
    """Evaluates a given model by returning its ROC AUC score, Cross Entropy loss, and Average Precision 
    score for both train and validation datasets."""
    
    # Predictions and probabilities
    train_predictions = model.predict_proba(X_train)[:,1]
    val_predictions = model.predict_proba(X_val)[:,1]
    
    # ROC AUC scores
    train_auc = roc_auc_score(Y_train, train_predictions)
    val_auc = roc_auc_score(Y_val, val_predictions)
    
    # Cross Entropy loss
    train_ce = log_loss(Y_train, train_predictions)
    val_ce = log_loss(Y_val, val_predictions)
    
    # Average Precision score
    train_ap = average_precision_score(Y_train, train_predictions)
    val_ap = average_precision_score(Y_val, val_predictions)
    
    return train_auc, val_auc, train_ce, val_ce, train_ap, val_ap
```


```python
n_trees_list = [5, 25, 50, 100]

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators = n_trees)
    rf.fit(X_train[numerical_cols], Y_train)
    
    train_auc, val_auc, train_ce, val_ce, train_ap, val_ap = evaluate_configuration(rf, X_train[numerical_cols], Y_train, 
                                                                                    X_val[numerical_cols], Y_val)
    print(f"For {n_trees} trees:")
    print(f"Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
    print(f"Train Cross Entropy: {train_ce:.4f}, Validation Cross Entropy: {val_ce:.4f}")
    print(f"Train Average Precision: {train_ap:.4f}, Validation Average Precision: {val_ap:.4f}")
    print()
    
```

    For 5 trees:
    Train AUC: 0.9885, Validation AUC: 0.6201
    Train Cross Entropy: 0.0273, Validation Cross Entropy: 0.3661
    Train Average Precision: 0.7240, Validation Average Precision: 0.0510
    
    For 25 trees:
    Train AUC: 0.9939, Validation AUC: 0.6893
    Train Cross Entropy: 0.0232, Validation Cross Entropy: 0.2671
    Train Average Precision: 0.8149, Validation Average Precision: 0.0794
    
    For 50 trees:
    Train AUC: 0.9943, Validation AUC: 0.7102
    Train Cross Entropy: 0.0231, Validation Cross Entropy: 0.2324
    Train Average Precision: 0.8239, Validation Average Precision: 0.0865
    
    For 100 trees:
    Train AUC: 0.9945, Validation AUC: 0.7313
    Train Cross Entropy: 0.0230, Validation Cross Entropy: 0.1963
    Train Average Precision: 0.8275, Validation Average Precision: 0.0913
    


We can observe that, as the number of trees increases, the model's performance on the validation set consistently improves. With only 5 trees, the model overfits the training data, evidenced by high training AUC and low validation AUC, alongside with high validation Cross Entropy and poor validation Average Precision. Increasing the number of trees to 25, 50, and 100 results in progressively better generalization, as indicated by rising validation AUC and Average Precision scores, and decreasing validation Cross Entropy loss. This trend suggests that using a larger number of trees enhances the model's ability to generalize from the training data to unseen validation data, with the best overall performance observed at 100 trees.

It is worth mentioning that the high variance is probably due to the fact that there is correlation between features, and therefore correlation between trees. Additionally, it is worth remembering that, in the Random Forest training, it is normal to have an AUC of 100%, as trees are constructed until the end and therefore each point arrives at a node. Nevertheless, in our case, it is not exactly 100%, as there is some node that has an instance of another class. In other words, there are samples with same input but different output, which indicates that there is either noise or we need more features (additional features may make previous equal instances different).

Now, we plot a feature importance plot to drop the lowest importance variables and see how the model performs.


```python
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train[numerical_cols], Y_train)
```




<style>#sk-container-id-1 {
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

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
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

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
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

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
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

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
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

#sk-container-id-1 a.estimator_doc_link {
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

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier()</pre></div> </div></div></div></div>




```python
def do_feature_importance(model, X_train):
    # Get feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis() 
    plt.show()

```


```python
do_feature_importance(rf, X_train[numerical_cols])
```


    
![png](Module_4_files/Module_4_43_0.png)
    


We now select only the features with highest importance:


```python
selected_cols = ['global_popularity', 'days_since_purchase_product_type', 'user_order_seq', 'days_since_purchase_variant_id', 
                 'std_days_to_buy_variant_id', 'avg_days_to_buy_variant_id', 'discount_pct', 'normalised_price']

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators = n_trees)
    rf.fit(X_train[selected_cols], Y_train)
    
    train_auc, val_auc, train_ce, val_ce, train_ap, val_ap = evaluate_configuration(rf, X_train[selected_cols], Y_train, 
                                                                                    X_val[selected_cols], Y_val)
    print(f"For {n_trees} trees:")
    print(f"Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
    print(f"Train Cross Entropy: {train_ce:.4f}, Validation Cross Entropy: {val_ce:.4f}")
    print(f"Train Average Precision: {train_ap:.4f}, Validation Average Precision: {val_ap:.4f}")
    print()
```

    For 5 trees:
    Train AUC: 0.9879, Validation AUC: 0.6227
    Train Cross Entropy: 0.0279, Validation Cross Entropy: 0.3607
    Train Average Precision: 0.7056, Validation Average Precision: 0.0538
    
    For 25 trees:
    Train AUC: 0.9933, Validation AUC: 0.6964
    Train Cross Entropy: 0.0240, Validation Cross Entropy: 0.2553
    Train Average Precision: 0.7969, Validation Average Precision: 0.0837
    
    For 50 trees:
    Train AUC: 0.9937, Validation AUC: 0.7172
    Train Cross Entropy: 0.0239, Validation Cross Entropy: 0.2202
    Train Average Precision: 0.8053, Validation Average Precision: 0.0891
    
    For 100 trees:
    Train AUC: 0.9939, Validation AUC: 0.7387
    Train Cross Entropy: 0.0238, Validation Cross Entropy: 0.1829
    Train Average Precision: 0.8090, Validation Average Precision: 0.0919
    


Once again, we can observe that as the number of trees increases, the model's performance on the validation set consistently improves. Increasing the number of trees results in progressively better generalization, as indicated by rising validation AUC and Average Precision scores, and decreasing validation Cross Entropy loss. 

We also have still high variance due to the fact that there is correlation between features, and therefore correlation between trees. Additionally, we don't have an AUC of exactly 100%, as there are samples with same input but different output, which indicates that there is either noise or we need more features (additional features may make previous equal instances different).

We can observe that reducing columns has increased the performance (validation with increased AUC, decreased Cross Entropy and increased validation precision), has this has probably made the correlation between trees decrease, and so they are more independent. Finally, the results are worse than those with Logistic Regression.

#### 4.2 Gradient Boosting tress
Gradient Boostring trees are constructed by taking the mean of the data and iteratively computing the gradient of the loss (predicting it using a new tree). Furthermore, their hyperparameters are: number of trees, learning rate, tree depth and regularisation terms of the loss.


```python
n_trees_list = [5, 25, 50, 100]
lrs = [0.05, 0.1]
tree_depths = [1, 3, 5]

for lr in lrs:
    for depth in tree_depths:
        for n_trees in n_trees_list:
            gb = GradientBoostingClassifier(n_estimators = n_trees, learning_rate = lr, max_depth = depth)
            gb.fit(X_train[numerical_cols], Y_train)
            
            train_auc, val_auc, train_ce, val_ce, train_ap, val_ap = evaluate_configuration(gb, X_train[numerical_cols], Y_train, 
                                                                                            X_val[numerical_cols], Y_val)
            print(f"For a learning rate of {lr}, a depth of {depth}, and {n_trees} trees,:")
            print(f"Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
            print(f"Train Cross Entropy: {train_ce:.4f}, Validation Cross Entropy: {val_ce:.4f}")
            print(f"Train Average Precision: {train_ap:.4f}, Validation Average Precision: {val_ap:.4f}")
            print()
    
```

    For a learning rate of 0.05, a depth of 1, and 5 trees,:
    Train AUC: 0.6937, Validation AUC: 0.6921
    Train Cross Entropy: 0.0756, Validation Cross Entropy: 0.0686
    Train Average Precision: 0.0542, Validation Average Precision: 0.0737
    
    For a learning rate of 0.05, a depth of 1, and 25 trees,:
    Train AUC: 0.7832, Validation AUC: 0.7907
    Train Cross Entropy: 0.0712, Validation Cross Entropy: 0.0636
    Train Average Precision: 0.0861, Validation Average Precision: 0.1226
    
    For a learning rate of 0.05, a depth of 1, and 50 trees,:
    Train AUC: 0.8024, Validation AUC: 0.8157
    Train Cross Entropy: 0.0691, Validation Cross Entropy: 0.0613
    Train Average Precision: 0.0976, Validation Average Precision: 0.1360
    
    For a learning rate of 0.05, a depth of 1, and 100 trees,:
    Train AUC: 0.8117, Validation AUC: 0.8265
    Train Cross Entropy: 0.0675, Validation Cross Entropy: 0.0595
    Train Average Precision: 0.1046, Validation Average Precision: 0.1436
    
    For a learning rate of 0.05, a depth of 3, and 5 trees,:
    Train AUC: 0.7487, Validation AUC: 0.7575
    Train Cross Entropy: 0.0731, Validation Cross Entropy: 0.0653
    Train Average Precision: 0.0879, Validation Average Precision: 0.1289
    
    For a learning rate of 0.05, a depth of 3, and 25 trees,:
    Train AUC: 0.8133, Validation AUC: 0.8282
    Train Cross Entropy: 0.0682, Validation Cross Entropy: 0.0602
    Train Average Precision: 0.1114, Validation Average Precision: 0.1541
    
    For a learning rate of 0.05, a depth of 3, and 50 trees,:
    Train AUC: 0.8142, Validation AUC: 0.8293
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1151, Validation Average Precision: 0.1564
    
    For a learning rate of 0.05, a depth of 3, and 100 trees,:
    Train AUC: 0.8172, Validation AUC: 0.8321
    Train Cross Entropy: 0.0659, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1211, Validation Average Precision: 0.1597
    
    For a learning rate of 0.05, a depth of 5, and 5 trees,:
    Train AUC: 0.8125, Validation AUC: 0.8273
    Train Cross Entropy: 0.0720, Validation Cross Entropy: 0.0643
    Train Average Precision: 0.1148, Validation Average Precision: 0.1540
    
    For a learning rate of 0.05, a depth of 5, and 25 trees,:
    Train AUC: 0.8156, Validation AUC: 0.8301
    Train Cross Entropy: 0.0670, Validation Cross Entropy: 0.0591
    Train Average Precision: 0.1282, Validation Average Precision: 0.1659
    
    For a learning rate of 0.05, a depth of 5, and 50 trees,:
    Train AUC: 0.8180, Validation AUC: 0.8318
    Train Cross Entropy: 0.0657, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1343, Validation Average Precision: 0.1670
    
    For a learning rate of 0.05, a depth of 5, and 100 trees,:
    Train AUC: 0.8207, Validation AUC: 0.8332
    Train Cross Entropy: 0.0651, Validation Cross Entropy: 0.0579
    Train Average Precision: 0.1424, Validation Average Precision: 0.1609
    
    For a learning rate of 0.1, a depth of 1, and 5 trees,:
    Train AUC: 0.7323, Validation AUC: 0.7431
    Train Cross Entropy: 0.0735, Validation Cross Entropy: 0.0660
    Train Average Precision: 0.0677, Validation Average Precision: 0.0991
    
    For a learning rate of 0.1, a depth of 1, and 25 trees,:
    Train AUC: 0.8021, Validation AUC: 0.8155
    Train Cross Entropy: 0.0689, Validation Cross Entropy: 0.0611
    Train Average Precision: 0.0966, Validation Average Precision: 0.1340
    
    For a learning rate of 0.1, a depth of 1, and 50 trees,:
    Train AUC: 0.8114, Validation AUC: 0.8263
    Train Cross Entropy: 0.0673, Validation Cross Entropy: 0.0594
    Train Average Precision: 0.1034, Validation Average Precision: 0.1418
    
    For a learning rate of 0.1, a depth of 1, and 100 trees,:
    Train AUC: 0.8146, Validation AUC: 0.8299
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1103, Validation Average Precision: 0.1496
    
    For a learning rate of 0.1, a depth of 3, and 5 trees,:
    Train AUC: 0.8077, Validation AUC: 0.8225
    Train Cross Entropy: 0.0707, Validation Cross Entropy: 0.0627
    Train Average Precision: 0.0984, Validation Average Precision: 0.1398
    
    For a learning rate of 0.1, a depth of 3, and 25 trees,:
    Train AUC: 0.8143, Validation AUC: 0.8295
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1136, Validation Average Precision: 0.1530
    
    For a learning rate of 0.1, a depth of 3, and 50 trees,:
    Train AUC: 0.8168, Validation AUC: 0.8319
    Train Cross Entropy: 0.0660, Validation Cross Entropy: 0.0581
    Train Average Precision: 0.1195, Validation Average Precision: 0.1562
    
    For a learning rate of 0.1, a depth of 3, and 100 trees,:
    Train AUC: 0.8187, Validation AUC: 0.8328
    Train Cross Entropy: 0.0656, Validation Cross Entropy: 0.0582
    Train Average Precision: 0.1262, Validation Average Precision: 0.1463
    
    For a learning rate of 0.1, a depth of 5, and 5 trees,:
    Train AUC: 0.8131, Validation AUC: 0.8276
    Train Cross Entropy: 0.0695, Validation Cross Entropy: 0.0616
    Train Average Precision: 0.1183, Validation Average Precision: 0.1568
    
    For a learning rate of 0.1, a depth of 5, and 25 trees,:
    Train AUC: 0.8185, Validation AUC: 0.8319
    Train Cross Entropy: 0.0657, Validation Cross Entropy: 0.0581
    Train Average Precision: 0.1321, Validation Average Precision: 0.1634
    
    For a learning rate of 0.1, a depth of 5, and 50 trees,:
    Train AUC: 0.8209, Validation AUC: 0.8332
    Train Cross Entropy: 0.0651, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1401, Validation Average Precision: 0.1547
    
    For a learning rate of 0.1, a depth of 5, and 100 trees,:
    Train AUC: 0.8225, Validation AUC: 0.8325
    Train Cross Entropy: 0.0646, Validation Cross Entropy: 0.0584
    Train Average Precision: 0.1516, Validation Average Precision: 0.1478
    


We select the set of hyperparameters that make the model perform the best and do feature importance. The combination that provides the highest validation AUC, the lowest validation Cross Entropy, and the highes validation Average Precision, indicating it has the best overall performance, is: learning rate: 0.05, depth: 5, and 100 trees.



```python
lr_best = 0.05
tree_depth_best = 5
n_trees_best = 100

gb = GradientBoostingClassifier(n_estimators = n_trees_best, learning_rate = lr_best, max_depth = tree_depth_best)
gb.fit(X_train[numerical_cols], Y_train)
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
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingClassifier(learning_rate=0.05, max_depth=5)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GradientBoostingClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">?<span>Documentation for GradientBoostingClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingClassifier(learning_rate=0.05, max_depth=5)</pre></div> </div></div></div></div>




```python
do_feature_importance(gb, X_train[numerical_cols])
```


    
![png](Module_4_files/Module_4_51_0.png)
    



```python
selected_cols = ['days_since_purchase_product_type', 'global_popularity', 'std_days_to_buy_product_type', 
                 'avg_days_to_buy_variant_id', 'days_since_purchase_product_type', 'std_days_to_buy_variant_id', 
                 'normalised_price', 'user_order_seq', 'avg_days_to_buy_product_type']

for lr in lrs:
    for depth in tree_depths:
        for n_trees in n_trees_list:
            gb = GradientBoostingClassifier(n_estimators = n_trees, learning_rate = lr, max_depth = depth)
            gb.fit(X_train[numerical_cols], Y_train)
            
            train_auc, val_auc, train_ce, val_ce, train_ap, val_ap = evaluate_configuration(gb, X_train[numerical_cols], Y_train, 
                                                                                            X_val[numerical_cols], Y_val)
            print(f"For a learning rate of {lr}, a depth of {depth}, and {n_trees} trees,:")
            print(f"Train AUC: {train_auc:.4f}, Validation AUC: {val_auc:.4f}")
            print(f"Train Cross Entropy: {train_ce:.4f}, Validation Cross Entropy: {val_ce:.4f}")
            print(f"Train Average Precision: {train_ap:.4f}, Validation Average Precision: {val_ap:.4f}")
            print()
    
```

    For a learning rate of 0.05, a depth of 1, and 5 trees,:
    Train AUC: 0.6937, Validation AUC: 0.6921
    Train Cross Entropy: 0.0756, Validation Cross Entropy: 0.0686
    Train Average Precision: 0.0542, Validation Average Precision: 0.0737
    
    For a learning rate of 0.05, a depth of 1, and 25 trees,:
    Train AUC: 0.7832, Validation AUC: 0.7907
    Train Cross Entropy: 0.0712, Validation Cross Entropy: 0.0636
    Train Average Precision: 0.0861, Validation Average Precision: 0.1226
    
    For a learning rate of 0.05, a depth of 1, and 50 trees,:
    Train AUC: 0.8024, Validation AUC: 0.8157
    Train Cross Entropy: 0.0691, Validation Cross Entropy: 0.0613
    Train Average Precision: 0.0976, Validation Average Precision: 0.1360
    
    For a learning rate of 0.05, a depth of 1, and 100 trees,:
    Train AUC: 0.8117, Validation AUC: 0.8265
    Train Cross Entropy: 0.0675, Validation Cross Entropy: 0.0595
    Train Average Precision: 0.1046, Validation Average Precision: 0.1436
    
    For a learning rate of 0.05, a depth of 3, and 5 trees,:
    Train AUC: 0.7487, Validation AUC: 0.7575
    Train Cross Entropy: 0.0731, Validation Cross Entropy: 0.0653
    Train Average Precision: 0.0879, Validation Average Precision: 0.1289
    
    For a learning rate of 0.05, a depth of 3, and 25 trees,:
    Train AUC: 0.8133, Validation AUC: 0.8282
    Train Cross Entropy: 0.0682, Validation Cross Entropy: 0.0602
    Train Average Precision: 0.1114, Validation Average Precision: 0.1541
    
    For a learning rate of 0.05, a depth of 3, and 50 trees,:
    Train AUC: 0.8142, Validation AUC: 0.8293
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1151, Validation Average Precision: 0.1564
    
    For a learning rate of 0.05, a depth of 3, and 100 trees,:
    Train AUC: 0.8172, Validation AUC: 0.8321
    Train Cross Entropy: 0.0659, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1211, Validation Average Precision: 0.1597
    
    For a learning rate of 0.05, a depth of 5, and 5 trees,:
    Train AUC: 0.8125, Validation AUC: 0.8273
    Train Cross Entropy: 0.0720, Validation Cross Entropy: 0.0643
    Train Average Precision: 0.1148, Validation Average Precision: 0.1540
    
    For a learning rate of 0.05, a depth of 5, and 25 trees,:
    Train AUC: 0.8156, Validation AUC: 0.8301
    Train Cross Entropy: 0.0670, Validation Cross Entropy: 0.0591
    Train Average Precision: 0.1282, Validation Average Precision: 0.1659
    
    For a learning rate of 0.05, a depth of 5, and 50 trees,:
    Train AUC: 0.8180, Validation AUC: 0.8318
    Train Cross Entropy: 0.0657, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1343, Validation Average Precision: 0.1670
    
    For a learning rate of 0.05, a depth of 5, and 100 trees,:
    Train AUC: 0.8207, Validation AUC: 0.8332
    Train Cross Entropy: 0.0651, Validation Cross Entropy: 0.0579
    Train Average Precision: 0.1424, Validation Average Precision: 0.1610
    
    For a learning rate of 0.1, a depth of 1, and 5 trees,:
    Train AUC: 0.7323, Validation AUC: 0.7431
    Train Cross Entropy: 0.0735, Validation Cross Entropy: 0.0660
    Train Average Precision: 0.0677, Validation Average Precision: 0.0991
    
    For a learning rate of 0.1, a depth of 1, and 25 trees,:
    Train AUC: 0.8021, Validation AUC: 0.8155
    Train Cross Entropy: 0.0689, Validation Cross Entropy: 0.0611
    Train Average Precision: 0.0966, Validation Average Precision: 0.1340
    
    For a learning rate of 0.1, a depth of 1, and 50 trees,:
    Train AUC: 0.8114, Validation AUC: 0.8263
    Train Cross Entropy: 0.0673, Validation Cross Entropy: 0.0594
    Train Average Precision: 0.1034, Validation Average Precision: 0.1418
    
    For a learning rate of 0.1, a depth of 1, and 100 trees,:
    Train AUC: 0.8146, Validation AUC: 0.8299
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1103, Validation Average Precision: 0.1496
    
    For a learning rate of 0.1, a depth of 3, and 5 trees,:
    Train AUC: 0.8077, Validation AUC: 0.8225
    Train Cross Entropy: 0.0707, Validation Cross Entropy: 0.0627
    Train Average Precision: 0.0984, Validation Average Precision: 0.1398
    
    For a learning rate of 0.1, a depth of 3, and 25 trees,:
    Train AUC: 0.8143, Validation AUC: 0.8295
    Train Cross Entropy: 0.0666, Validation Cross Entropy: 0.0586
    Train Average Precision: 0.1136, Validation Average Precision: 0.1530
    
    For a learning rate of 0.1, a depth of 3, and 50 trees,:
    Train AUC: 0.8168, Validation AUC: 0.8319
    Train Cross Entropy: 0.0660, Validation Cross Entropy: 0.0581
    Train Average Precision: 0.1195, Validation Average Precision: 0.1562
    
    For a learning rate of 0.1, a depth of 3, and 100 trees,:
    Train AUC: 0.8187, Validation AUC: 0.8328
    Train Cross Entropy: 0.0656, Validation Cross Entropy: 0.0582
    Train Average Precision: 0.1262, Validation Average Precision: 0.1464
    
    For a learning rate of 0.1, a depth of 5, and 5 trees,:
    Train AUC: 0.8131, Validation AUC: 0.8276
    Train Cross Entropy: 0.0695, Validation Cross Entropy: 0.0616
    Train Average Precision: 0.1183, Validation Average Precision: 0.1568
    
    For a learning rate of 0.1, a depth of 5, and 25 trees,:
    Train AUC: 0.8185, Validation AUC: 0.8319
    Train Cross Entropy: 0.0657, Validation Cross Entropy: 0.0581
    Train Average Precision: 0.1321, Validation Average Precision: 0.1635
    
    For a learning rate of 0.1, a depth of 5, and 50 trees,:
    Train AUC: 0.8209, Validation AUC: 0.8332
    Train Cross Entropy: 0.0651, Validation Cross Entropy: 0.0580
    Train Average Precision: 0.1401, Validation Average Precision: 0.1545
    
    For a learning rate of 0.1, a depth of 5, and 100 trees,:
    Train AUC: 0.8225, Validation AUC: 0.8325
    Train Cross Entropy: 0.0646, Validation Cross Entropy: 0.0584
    Train Average Precision: 0.1516, Validation Average Precision: 0.1479
    


The set of hyperparameters that achieves one of the highest AUCs, lowest Cross Entropies, and the highest Average Precision is learning rate of 0.05, depth 5 and 50 trees. Therefore, it represents the best balance among the three metrics.

### 5. Comparing selected models


```python
# Logistic Regression
reduced_cols = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', C=1e-4)
)

lr.fit(X_train[reduced_cols], Y_train)
lr_predictions = lr.predict_proba(X_test[reduced_cols])[:, 1]


# Random Forest
selected_cols = ['global_popularity', 'days_since_purchase_product_type', 'user_order_seq', 'days_since_purchase_variant_id', 
                 'std_days_to_buy_variant_id', 'avg_days_to_buy_variant_id', 'discount_pct', 'normalised_price', 
                 'std_days_to_buy_product_type', 'avg_days_to_buy_product_type', 'count_pets']

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train[selected_cols], Y_train)
rf_predictions = rf.predict_proba(X_test[selected_cols])[:, 1]


# Gradient Boosting
selected_cols = ['days_since_purchase_product_type', 'global_popularity', 'std_days_to_buy_product_type', 
                 'avg_days_to_buy_variant_id', 'days_since_purchase_product_type', 'std_days_to_buy_variant_id', 
                 'normalised_price', 'user_order_seq', 'avg_days_to_buy_product_type']

gb = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.05, max_depth = 5)
gb.fit(X_train[selected_cols], Y_train)
gb_predictions = gb.predict_proba(X_test[selected_cols])[:, 1]
```


```python
def compute_metrics(y_true, predictions):
    metrics = {}
    for model_name, preds in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_score = auc(fpr, tpr)
        ap_score = average_precision_score(y_true, preds)
        
        metrics[model_name] = {
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score,
            'ap': ap_score
        }
    
    return metrics

```


```python
def plot_metrics(metrics):
    # Plot Precision-Recall curves
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for model_name, metric in metrics.items():
        plt.plot(metric['recall'], metric['precision'], label=f'{model_name} (AP={metric["ap"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Plot ROC curves
    plt.subplot(1, 2, 2)
    for model_name, metric in metrics.items():
        plt.plot(metric['fpr'], metric['tpr'], label=f'{model_name} (AUC={metric["auc"]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

```


```python
# Example usage with your predictions
predictions = {
    'Logistic Regression': lr_predictions,
    'Random Forest': rf_predictions,
    'Gradient Boosting': gb_predictions
}

metrics = compute_metrics(Y_test, predictions)
plot_metrics(metrics)
```


    
![png](Module_4_files/Module_4_58_0.png)
    



    The Kernel crashed while executing code in the current cell or a previous cell. 


    Please review the code in the cell(s) to identify a possible cause of the failure. 


    Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 


    View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.


We can osberve that in the Precision-Recall Curve, the Logistic Regression has the highest average precision (AP),compared to Random Forest and Gradient Boosting. Furthermore, Random Forest has a moderate AP, while Gradient Boosting has the lowest AP, indicating weaker performance in distinguishing the positive class.

On the other hand, in the ROC Curve, Logistic Regression and Gradient Boosting show similar performance with AUCs of 0.81 and 0.80, respectively. Random Forest performs worse with an AUC of 0.73, indicating a higher rate of false positives and/or false negatives compared to the other two models.

In conclusion, Logistic Regression is the best performer in terms of both Precision-Recall and ROC metrics; Gradient Boosting has comparable ROC performance to Logistic Regression but performs worse in the Precision-Recall metric; and, finally, Random Forest has the lowest performance in both Precision-Recall and ROC metrics.
