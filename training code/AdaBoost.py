import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import AdaBoostClassifier
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

# === إنشاء الميزات ===
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

# === Calculate feature thresholds ===
feature_thresholds = {}
for feature in feature_order:
    mean = X_train[feature].mean()
    std = X_train[feature].std()
    threshold = mean + 2 * std
    feature_thresholds[feature] = float(threshold)

# === Train AdaBoost model ===
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"\n✅ AdaBoost Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Confusion Matrix:")
print(confusion_mat)
print("\n📝 Classification Report:")
print(classification_rep)

# === ROC Curve ===
y_scores = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (AdaBoost)")
plt.grid()
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_adaboost.png'))
plt.close()

# === Save model and metadata ===
model_file = os.path.join(MODEL_DIR, "adaboost_model.pkl")
scaler_file = os.path.join(MODEL_DIR, "adaboost_scaler.pkl")
joblib.dump(model, model_file)
joblib.dump(scaler, scaler_file)

metadata = {
    "adaboost_model": {
        "model_file": "adaboost_model.pkl",
        "scaler_file": "adaboost_scaler.pkl",
        "accuracy": float(accuracy),
        "threshold": 0.5,
        "feature_order": feature_order,
        "feature_thresholds": feature_thresholds
    }
}

with open(os.path.join(MODEL_DIR, "adaboost_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n✅ AdaBoost Model, Scaler, and Metadata saved successfully.")