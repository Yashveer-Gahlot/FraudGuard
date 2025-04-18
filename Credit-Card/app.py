from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Create Flask app with proper static folder configuration
app = Flask(__name__, 
            static_folder='static',  
            template_folder='templates')

# Load all models - with error handling
models = {}
model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'SVM': 'svm.pkl',
    'XGBoost': 'xgboost.pkl',
    'LightGBM': 'lightgbm.pkl'
}

# Load models with error handling
for model_name, file_name in model_files.items():
    try:
        if os.path.exists(file_name):
            models[model_name] = pickle.load(open(file_name, 'rb'))
            print(f"Loaded {model_name} successfully")
        else:
            print(f"Warning: Model file {file_name} not found")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

# Load scaler if available
scaler = None
try:
    if os.path.exists('scaler.pkl'):
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        print("Scaler loaded successfully")
    else:
        print("Warning: scaler.pkl not found")
except Exception as e:
    print(f"Error loading scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        form_data = request.form.to_dict()
        model_name = form_data.pop("model")
        
        # Check if selected model is available
        if model_name not in models:
            return render_template('error.html', message=f"Model {model_name} is not available")
        
        try:
            # Extract Time and Amount only - the model seems to expect only these features
            try:
                time_val = float(form_data['Time'])
                amount_val = float(form_data['Amount'])
            except ValueError:
                return render_template('error.html', message="Invalid Time or Amount value. Please enter numbers only.")
            except KeyError:
                return render_template('error.html', message="Missing Time or Amount values.")
            
            # Apply scaling to Time and Amount if scaler is available
            if scaler is not None:
                scaled_values = scaler.transform([[time_val, amount_val]])
                time_scaled = scaled_values[0][0]
                amount_scaled = scaled_values[0][1]
            else:
                # If no scaler, use the raw values (not ideal but better than failing)
                time_scaled = time_val
                amount_scaled = amount_val
            
            # Create input array with just Time and Amount (2 features)
            input_array = np.array([time_scaled, amount_scaled]).reshape(1, -1)
            
            # Make prediction
            selected_model = models[model_name]
            prediction = selected_model.predict(input_array)[0]
            
            # Get probability if available
            probability = None
            if hasattr(selected_model, "predict_proba"):
                probability = selected_model.predict_proba(input_array)[0][1]
                probability_pct = f"{probability:.2%}"
            else:
                probability_pct = "N/A"
            
            result_text = "⚠️ Fraudulent Transaction Detected!" if prediction == 1 else "✅ Transaction is Not Fraudulent"
            return render_template('result.html', 
                                  prediction=result_text, 
                                  model=model_name,
                                  probability=probability_pct)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return render_template('error.html', 
                                   message=f"Error during prediction: {str(e)}", 
                                   details=error_details)

# Error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)
    