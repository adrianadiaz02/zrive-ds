from sklearn.ensemble import GradientBoostingClassifier

class PushModel:
    FEATURE_COLUMNS = ["user_order_seq", "ordered_before", "abandoned_before", 
        "active_snoozed", "set_as_regular", "normalised_price", "discount_pct", 
        "global_popularity", "count_adults", "count_children", "count_babies", "count_pets", 
        "people_ex_baby", "days_since_purchase_variant_id", "avg_days_to_buy_variant_id", 
        "std_days_to_buy_variant_id", "days_since_purchase_product_type", 
        "avg_days_to_buy_product_type", "std_days_to_buy_product_type"]
        
    TARGET_COLUMN = "outcome"

    def __init__(self, hyperparameters) -> None:
        self.gbc = GradientBoostingClassifier(**hyperparameters)


    def extract_features(self, df):
        """Given a dataframe, returns a dataframe that contains only the feature columns (defined in FEATURE_COLUMNS)"""
        return df[self.FEATURE_COLUMNS]


    def extract_target(self, df):
        """Given a dataframe, returns only the target/label column ("outcome")"""
        return df[self.TARGET_COLUMN]


    def feature_target_split(self, df):
        """Given a dataframe, divides the input in two subsets:
        - a dataframe with the features subset
        - a series with the target column"""
        X = self.extract_features(df)
        Y = self.extract_target(df)
        return X, Y


    def fit_model(self, df):
        """Given a complete dataframe (with both features and target columns), fits a Gradient Boosting model"""
        features, label = self.feature_target_split(df)
        self.gbc.fit(features, label)
        return self.gbc

