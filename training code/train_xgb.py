import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# === إعداد المسارات ===
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === تحميل البيانات ===
df = pd.read_csv(r"C:\Users\mstfy\Downloads\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.columns = df.columns.str.strip()

# === إنشاء الميزات (تتماشى مع cli1.py) ===
df['flow_rate'] = df['Flow Packets/s']
df['syn_rate'] = df['SYN Flag Count'] / df['Flow Duration'].replace(0, 0.001)
df['avg_packet_size'] = df['Average Packet Size']
df['syn_ratio'] = df['SYN Flag Count'] / df['Total Fwd Packets'].replace(0, 1)
df['ack_ratio'] = df['ACK Flag Count'] / df['Total Fwd Packets'].replace(0, 1)
df['psh_ratio'] = df['PSH Flag Count'] / df['Total Fwd Packets'].replace(0, 1)
df['urg_ratio'] = df['URG Flag Count'] / df['Total Fwd Packets'].replace(0, 1)
df['flow_duration'] = df['Flow Duration']
df['down_up_ratio'] = df['Down/Up Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
df['entropy'] = df['Packet Length Std'].replace(np.nan, 0)

# === تحديد الأعمدة المستهدفة ===
feature_order = [
    'flow_rate', 'syn_rate', 'avg_packet_size', 'syn_ratio', 'ack_ratio',
    'psh_ratio', 'urg_ratio', 'entropy', 'flow_duration', 'down_up_ratio'
]
X = df[feature_order]
y = df['Label'].apply(lambda x: 1 if str(x).strip().lower() == 'ddos' else 0)

# === تنظيف البيانات ===
X.replace([np.inf, -np.inf], 0, inplace=True)
X.fillna(0, inplace=True)

# === تقسيم البيانات ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === التحجيم ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === تدريب النموذج الأول (Random Forest) ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# === التقييم ===
rf_y_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

print(f"\n✅ Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
print("\n📊 Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_y_pred))
print("\n📝 Classification Report (Random Forest):")
print(classification_report(y_test, rf_y_pred))

# === تدريب النموذج الثاني (Gradient Boosting) ===
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# === التقييم ===
gb_y_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_y_pred)

print(f"\n✅ Gradient Boosting Model Accuracy: {gb_accuracy * 100:.2f}%")
print("\n📊 Confusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_test, gb_y_pred))
print("\n📝 Classification Report (Gradient Boosting):")
print(classification_report(y_test, gb_y_pred))

# === رسم منحنى ROC للنموذج الأول (Random Forest) ===
rf_y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_y_scores)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, tpr, label="TPR (True Positive Rate)")
plt.plot(thresholds, 1 - fpr, label="1 - FPR (True Negative Rate)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("ROC Threshold Tuning (Random Forest)")
plt.grid()
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_rf.png'))
plt.close()

# === حساب عتبات الميزات (Feature Thresholds) ===
feature_thresholds = {}
for feature in feature_order:
    mean = X_train[feature].mean()
    std = X_train[feature].std()
    threshold = mean + 2 * std
    feature_thresholds[feature] = float(threshold) if not np.isnan(threshold) else 0.0

# === الحفظ ===
rf_model_file = os.path.join(MODEL_DIR, "random_forest_model.pkl")
gb_model_file = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
scaler_file = os.path.join(MODEL_DIR, "default_scaler.pkl")

joblib.dump(rf_model, rf_model_file)
joblib.dump(gb_model, gb_model_file)
joblib.dump(scaler, scaler_file)

metadata = {
    "random_forest_model": {
        "model_file": "random_forest_model.pkl",
        "scaler_file": "default_scaler.pkl",
        "accuracy": float(rf_accuracy),
        "threshold": 0.5,
        "feature_order": feature_order,
        "feature_thresholds": feature_thresholds
    },
    "gradient_boosting_model": {
        "model_file": "gradient_boosting_model.pkl",
        "scaler_file": "default_scaler.pkl",
        "accuracy": float(gb_accuracy),
        "threshold": 0.5,
        "feature_order": feature_order,
        "feature_thresholds": feature_thresholds
    }
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Models, Scaler, and Metadata saved successfully.")
print(f"Feature Thresholds: {feature_thresholds}")