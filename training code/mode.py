import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
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

# === تدريب النماذج ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}
for model_name, model in models.items():
    print(f"\n🔄 Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    print(f"✅ {model_name} Accuracy: {accuracy * 100:.2f}%")

# === حفظ النماذج ===
metadata = {}
for model_name, result in results.items():
    model_file = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(result["model"], model_file)
    metadata[model_name] = {
        "model_file": model_file,
        "accuracy": result["accuracy"],
        "feature_order": feature_order
    }

# === حفظ البيانات الوصفية ===
with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

# === عرض النتائج ===
best_model = max(results, key=lambda x: results[x]["accuracy"])
print(f"\n🏆 Best Model: {best_model} with Accuracy: {results[best_model]['accuracy'] * 100:.2f}%")
print("\n📊 Detailed Results:")
for model_name, result in results.items():
    print(f"\n🔍 {model_name}:")
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("Classification Report:")
    print(result["classification_report"])