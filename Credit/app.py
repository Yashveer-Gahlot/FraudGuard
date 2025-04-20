from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, 
            static_folder='static',  
            template_folder='templates')

# Load all models
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

# Load scaler
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        model_name = form_data.pop("model", None)

        if model_name not in models:
            return render_template('error.html', message=f"Model '{model_name}' is not available or not loaded.")

        try:
            # Get input values
            try:
                time_val = float(form_data['Time'])
                amount_val = float(form_data['Amount'])
            except ValueError:
                return render_template('error.html', message="Please enter valid numeric values for Time and Amount.")
            except KeyError:
                return render_template('error.html', message="Missing Time or Amount value.")

            # Scale the inputs
            if scaler:
                scaled_input = scaler.transform([[time_val, amount_val]])
            else:
                scaled_input = [[time_val, amount_val]]

            input_array = np.array(scaled_input).reshape(1, -1)

            # Make prediction
            selected_model = models[model_name]
            prediction = selected_model.predict(input_array)[0]

            probability = None
            if hasattr(selected_model, "predict_proba"):
                probability = selected_model.predict_proba(input_array)[0][1]
                # Only show one percentage sign
                probability_pct = f"{probability * 100:.2f}%"  # formatting to show one percentage sign
            else:
                probability_pct = "N/A"

            result_text = "\u26a0\ufe0f Fraudulent Transaction Detected!" if prediction == 1 else "\u2705 Transaction is Not Fraudulent"

            return render_template('result.html',
                                   prediction=result_text,
                                   model=model_name,
                                   probability=probability_pct)
        except Exception as e:
            import traceback
            return render_template('error.html', message=f"Error during prediction: {str(e)}",
                                   details=traceback.format_exc())
    else:
        return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)
    