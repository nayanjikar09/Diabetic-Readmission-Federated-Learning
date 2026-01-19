from flask import Flask, render_template, request, redirect, url_for
import os
import json
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# ---------------- Configuration ----------------
# Use os.getcwd() to ensure we are looking in the current working directory
BASE_DIR = os.path.abspath(os.getcwd())
CLIENT_WEIGHTS_DIR = os.path.join(BASE_DIR, "clients_weights")
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "global_model.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

# Ensure the upload directory exists
if not os.path.exists(CLIENT_WEIGHTS_DIR):
    os.makedirs(CLIENT_WEIGHTS_DIR, exist_ok=True)

# ---------------- History Logic ----------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {"round": 0, "accuracy": 0.0, "logs": []}
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            # Ensure required keys exist
            if "logs" not in data: data["logs"] = []
            return data
    except Exception:
        return {"round": 0, "accuracy": 0.0, "logs": []}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ---------------- Preprocessing ----------------
def preprocess(df):
    gender_map = {'M': 0, 'F': 1}
    diag_map = {'Diabetes': 0, 'Hypertension': 1, 'Heart Disease': 2, 'Asthma': 3, 'Pneumonia': 4, 'Infection': 5}
    discharge_map = {'Home': 0, 'Rehab': 1}

    df['Gender'] = df['Gender'].map(gender_map).fillna(0)
    df['Diagnosis'] = df['Diagnosis'].map(diag_map).fillna(0)
    df['Discharge_Type'] = df['Discharge_Type'].map(discharge_map).fillna(0)

    features = ["Age", "Gender", "Diagnosis", "Length_of_Stay", "Num_Prior_Admissions", "Discharge_Type"]
    return df[features].astype(float)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    history = load_history()
    # Check if folder exists before listing
    client_files = []
    if os.path.exists(CLIENT_WEIGHTS_DIR):
        client_files = [f for f in os.listdir(CLIENT_WEIGHTS_DIR) if f.endswith(".pkl")]
    return render_template("dashboard.html", history=history, clients=len(client_files))

@app.route("/reset")
def reset():
    """Wipe everything to fix 'str' or 'No valid model' errors."""
    if os.path.exists(GLOBAL_MODEL_PATH):
        os.remove(GLOBAL_MODEL_PATH)
    if os.path.exists(CLIENT_WEIGHTS_DIR):
        for f in os.listdir(CLIENT_WEIGHTS_DIR):
            os.remove(os.path.join(CLIENT_WEIGHTS_DIR, f))
    save_history({"round": 0, "accuracy": 0.0, "logs": []})
    return "System Reset! Upload fresh .pkl files and click Merge."

# ---------------- Prediction ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        return render_template("index.html", result="⚠️ Global model not trained yet")

    try:
        with open(GLOBAL_MODEL_PATH, "rb") as f:
            ensemble = pickle.load(f)
        
        models = ensemble["models"]
        weights = ensemble["weights"]

        df = pd.DataFrame([{
            "Age": int(request.form["age"]),
            "Gender": request.form["gender"],
            "Diagnosis": request.form["diagnosis"],
            "Length_of_Stay": int(request.form["los"]),
            "Num_Prior_Admissions": int(request.form["prior"]),
            "Discharge_Type": request.form["discharge"]
        }])

        X = preprocess(df)
        
        final_prob = 0
        valid_models_used = 0

        for model, w in zip(models, weights):
            # Check if it's a model object with the predict_proba method
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0][1]
                final_prob += (prob * w)
                valid_models_used += 1
        
        if valid_models_used == 0:
            return render_template("index.html", result="❌ Error: The global model is empty or corrupted.")

        prediction = 1 if final_prob >= 0.5 else 0
        return render_template("index.html", result="Readmitted" if prediction == 1 else "Not Readmitted")

    except Exception as e:
        print(f"DEBUG Error: {e}")
        return render_template("index.html", result=f"❌ Prediction error: {str(e)}")

# ---------------- Federated Aggregation ----------------
@app.route("/merge")
def merge():
    if not os.path.exists(CLIENT_WEIGHTS_DIR):
        return redirect(url_for("dashboard"))

    weight_files = [f for f in os.listdir(CLIENT_WEIGHTS_DIR) if f.endswith(".pkl")]
    
    if not weight_files:
        return "❌ Error: No .pkl files found in 'clients_weights' folder."

    models = []
    accuracies = []

    for wf in weight_files:
        model_path = os.path.join(CLIENT_WEIGHTS_DIR, wf)
        # Search for info in same directory
        info_path = model_path.replace("_weights.pkl", "_info.json").replace(".pkl", "_info.json")

        try:
            with open(model_path, "rb") as f:
                loaded_data = pickle.load(f)
                
                # Robust Unwrap
                if isinstance(loaded_data, list):
                    actual_model = loaded_data[0]
                else:
                    actual_model = loaded_data

                # Check if it's an actual model object (prevents 'str' errors)
                if hasattr(actual_model, "predict_proba"):
                    models.append(actual_model)
                    
                    # Try to find accuracy
                    acc = 50
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f_info:
                            info = json.load(f_info)
                            acc = info.get("accuracy", info.get("local_accuracy", 50))
                    accuracies.append(acc)
                else:
                    print(f"Skipping {wf}: It is a {type(actual_model)}, not a model.")

        except Exception as e:
            print(f"Failed to load {wf}: {e}")

    if not models:
        return f"❌ Error: Found files {weight_files}, but none were valid model objects."

    # Weighted Average
    total_acc = sum(accuracies) if sum(accuracies) > 0 else len(accuracies)
    normalized_weights = [acc / total_acc for acc in accuracies]

    # Save as Ensemble
    with open(GLOBAL_MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "weights": normalized_weights}, f)

    # Update history
    history = load_history()
    history["round"] += 1
    current_acc = round(float(np.mean(accuracies)), 2)
    history["accuracy"] = current_acc
    history["logs"].append({
        "round": history["round"],
        "clients": len(models),
        "accuracy": current_acc
    })
    save_history(history)

    # Cleanup client files after merging
    for f in os.listdir(CLIENT_WEIGHTS_DIR):
        try: os.remove(os.path.join(CLIENT_WEIGHTS_DIR, f))
        except: pass

    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)