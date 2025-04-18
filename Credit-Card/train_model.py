import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost and LightGBM with proper error handling
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    print("XGBoost not installed. Skipping XGBoost model.")
    xgboost_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    print("LightGBM not installed. Skipping LightGBM model.")
    lightgbm_available = False

# Create directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('creditcard.csv')
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: creditcard.csv file not found. Please ensure the file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Check for missing values
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"Warning: Dataset contains {missing_values} missing values")
    # Handle missing values - imputation or removal
    df = df.dropna()

# Balance the data
fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(n=len(fraud)*5, random_state=42)
data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

print(f"Using balanced dataset with {len(fraud)} fraudulent transactions and {len(non_fraud)} non-fraudulent transactions")

X = data.drop('Class', axis=1)
y = data['Class']

# Scale Time and Amount
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Save the scaler for later use in prediction
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    print("Scaler saved for preprocessing new data")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary of models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'decision_tree': DecisionTreeClassifier(class_weight='balanced'),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'svm': SVC(probability=True, class_weight='balanced')
}

# Add XGBoost and LightGBM if available
if xgboost_available:
    models['xgboost'] = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        scale_pos_weight=len(non_fraud)/len(fraud)  # Class balancing
    )

if lightgbm_available:
    models['lightgbm'] = lgb.LGBMClassifier(
        class_weight='balanced',
        verbose=-1  # Suppress output
    )

# Function to evaluate and save model performance
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # For ROC curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # If predict_proba not available, use decision_function if available
        if hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred  # Fallback
    
    # Calculate precision-recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n{name.upper()} EVALUATION:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'pr_auc': pr_auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

# Train, evaluate and save each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train, y_train)
        
        # Save model
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} trained and saved.")
        
        # Evaluate model
        results[name] = evaluate_model(name, model, X_test, y_test)
        
    except Exception as e:
        print(f"Error training {name}: {e}")

# Plot performance comparison
plt.figure(figsize=(10, 6))
accuracies = [results[model]['accuracy'] for model in results]
pr_aucs = [results[model]['pr_auc'] for model in results]

plt.bar(results.keys(), accuracies, alpha=0.7, label='Accuracy')
plt.bar(results.keys(), pr_aucs, alpha=0.7, label='PR-AUC')
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel comparison chart saved as 'model_comparison.png'")
