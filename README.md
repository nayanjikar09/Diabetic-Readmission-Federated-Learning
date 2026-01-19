![Dashboard Preview](landing_page.png)
🏥 Federated Learning for Diabetic Readmission
📝 Overview
This project demonstrates a Federated Learning (FL) approach to a healthcare classification problem. Using XGBoost, we predict whether a diabetic patient will be readmitted to the hospital within 30 days (<30), after 30 days (>30), or not at all (NO).

The core challenge addressed is Data Silos: Hospitals cannot share patient data due to HIPAA/GDPR regulations. This system allows models to learn from multiple hospitals (Clients A, B, and C) by sharing model parameters (weights) instead of raw data.

🌟 Key Features
Privacy-Preserving: Local patient data never leaves the hospital's infrastructure.

High Accuracy (75%+): Achieved through advanced hyperparameter tuning and cost-sensitive learning.

Handling Imbalance: Implements sample_weight to specifically improve detection of the critical <30 day readmission group.

Robust Preprocessing: Handles synthetic healthcare data anomalies, missing values, and complex categorical feature encoding.

🏗️ Technical Architecture
The system consists of two main components:

Client Nodes (Hospitals):

Each hospital runs a local XGBoost training session.

Uses RandomizedSearchCV for local optimization.

Exports a serialized .pkl weight file and a metadata .json file.

Central Server:

Acts as the aggregator.

Collects weights from all clients.

Performs Federated Averaging (FedAvg) or weighted aggregation to update the Global Model.

📊 Results & Evaluation
The model is evaluated using:

Accuracy Score: Targeted at >75%.

F1-Macro: Ensuring the minority class (<30) is correctly identified.

Confusion Matrix: Visualizing the precision-recall trade-off across all 3 classes.
