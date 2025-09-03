# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import shap

# --- CORRECTED DATA LOADING AND MERGING ---

print("Loading datasets...")
orders = pd.read_csv('data/olist_orders_dataset.csv')
sellers = pd.read_csv('data/olist_sellers_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
order_payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv') # Added for review score

print("Aggregating and merging data...")

# **FIX 1: Aggregate payments per order BEFORE merging**
order_payments_agg = order_payments.groupby('order_id')['payment_value'].sum().reset_index()

# **FIX 2: Aggregate reviews per order BEFORE merging**
order_reviews_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()

# Merge order_items with sellers
df = pd.merge(order_items, sellers, on='seller_id')
# Merge with orders to get timestamps
df = pd.merge(df, orders, on='order_id')
# Merge with the AGGREGATED payments
df = pd.merge(df, order_payments_agg, on='order_id')
# Merge with the AGGREGATED reviews
df = pd.merge(df, order_reviews_agg, on='order_id')

# Convert date columns to datetime objects
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Filter for relevant order statuses
df = df[df['order_status'] == 'delivered'].copy()

print("Data loaded and merged successfully!")

# --- FEATURE ENGINEERING (Your code here was correct) ---

print("Starting feature engineering...")
# Find the most recent purchase date in the entire dataset
latest_date = df['order_purchase_timestamp'].max()
churn_threshold_days = 90
df_churn = df.groupby('seller_id')['order_purchase_timestamp'].max().reset_index()
df_churn['days_since_last_order'] = (latest_date - df_churn['order_purchase_timestamp']).dt.days
df_churn['is_churned'] = (df_churn['days_since_last_order'] > churn_threshold_days).astype(int)

# 1. Recency, Frequency, Monetary (RFM)
rfm = df.groupby('seller_id').agg({
    'order_purchase_timestamp': lambda date: (latest_date - date.max()).days, # Recency
    'order_id': 'nunique', # Frequency
    'payment_value': 'sum' # Monetary
}).rename(columns={
    'order_purchase_timestamp': 'recency',
    'order_id': 'frequency',
    'payment_value': 'monetary'
})

# 2. Other Behavioral Features
behavioral = df.groupby('seller_id').agg({
    'review_score': 'mean',
    'product_id': 'nunique',
    'price': ['mean', 'std']
}).reset_index()
# Flatten multi-level column names
behavioral.columns = ['seller_id', 'avg_review_score', 'num_unique_products', 'avg_product_price', 'std_product_price']
behavioral.fillna(0, inplace=True) # Fill std deviation NaN for single-product sellers

# Combine all features into a final seller dataset
seller_features = pd.merge(rfm, behavioral, on='seller_id')
final_df = pd.merge(seller_features, df_churn[['seller_id', 'is_churned']], on='seller_id')

print("Feature engineering complete!")

# --- MODEL TRAINING AND SAVING (Your code here was correct) ---

print("Starting model training...")
X = final_df.drop(['seller_id', 'is_churned'], axis=1)
y = final_df['is_churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

print("\nClassification Report:")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

print("\nGenerating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("\nSaving model and other artifacts...")
joblib.dump(model, 'churn_model.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')
print("\nModel, explainer, and test data saved successfully!")