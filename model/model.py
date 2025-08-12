import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
import joblib  # สำหรับ save model

# 📁 สร้างโฟลเดอร์ผลลัพธ์
os.makedirs("results", exist_ok=True)

# 🔄 โหลดข้อมูล
df = pd.read_csv("voice_features.csv")
X = df.drop(['label', 'filename'], axis=1, errors='ignore').values
y = df['label']
if y.dtype == 'O':
    y = y.map({'PD': 1, 'Control': 0}).values

# 🎯 Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs, aucs, recalls, f1s = [], [], [], []
all_cm = np.zeros((2, 2))
best_f1 = 0
best_model = None

metrics_list = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=2,
        max_features='sqrt'
    )
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    accs.append(acc)
    aucs.append(auc)
    recalls.append(recall)
    f1s.append(f1)
    all_cm += cm

    metrics_list.append([fold + 1, acc, auc, recall, f1])
    print(f"Fold {fold+1}: acc={acc:.3f}, auc={auc:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 50)

    # 📌 Save best fold model
    if f1 > best_f1:
        best_f1 = f1
        best_model = clf

# 💾 Save best fold model
joblib.dump(best_model, "results/best_model.pkl")
print("💾 Saved best fold model to results/best_model.pkl")

# 📝 Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list, columns=["Fold", "Accuracy", "AUC", "Recall_PD", "F1_PD"])
metrics_df.to_csv("results/metrics_per_fold.csv", index=False)

# 📝 Save average metrics
with open("results/summary.txt", "w", encoding="utf-8") as f:
    f.write("==== เฉลี่ย 10-Fold ====\n")
    f.write(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")
    f.write(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")
    f.write(f"Recall (PD): {np.mean(recalls):.3f}\n")
    f.write(f"F1 (PD): {np.mean(f1s):.3f}\n")
    f.write("Confusion Matrix sum:\n")
    f.write(str(all_cm.astype(int)))

# 📊 Plot Confusion Matrix (แบบ integer เท่านั้น)
plt.figure(figsize=(6, 5))
sns.heatmap(all_cm.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Control", "PD"], yticklabels=["Control", "PD"])
plt.title("Confusion Matrix (10-Fold Sum)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()


# 📈 Plot Metrics over Folds
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), accs, marker='o', label='Accuracy')
plt.plot(range(1, 11), aucs, marker='o', label='AUC')
plt.plot(range(1, 11), recalls, marker='o', label='Recall (PD)')
plt.plot(range(1, 11), f1s, marker='o', label='F1 (PD)')
plt.xticks(range(1, 11))
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Metrics per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/metrics_per_fold.png")
plt.close()

# ✅ เทรนใหม่บนข้อมูลทั้งหมดเพื่อใช้งานจริง
final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    min_samples_leaf=2,
    max_features='sqrt'
)

# ⚙️ ทำ SMOTE กับข้อมูลทั้งหมด
sm = SMOTE(random_state=42)
X_full_res, y_full_res = sm.fit_resample(X, y)
final_model.fit(X_full_res, y_full_res)

# 💾 บันทึก final_model พร้อม feature_names
feature_names = df.drop(['label', 'filename'], axis=1, errors='ignore').columns.tolist()
model_package = {
    "model": final_model,
    "feature_names": feature_names
}
joblib.dump(model_package, "results/final_model_for_use.pkl")
print("✅ Saved final model (trained on all data) to results/final_model_for_use.pkl")

print("📊 All metrics, models, and plots saved in folder: results/")
