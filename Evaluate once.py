  import pandas as pd
import json
import pickle
from sklearn.metrics import confusion_matrix, classification_report

print("=== TASK 6: EVALUATE ONCE (TEST SET) ===")

df = pd.read_json('ecg_quality_features.jsonl', lines=True)
df['split'] = df['participant_id'].map(json.load(open('participant_split.json')))
frozen = json.load(open('frozen_method.json', 'r'))
models = pickle.load(open('models.pkl', 'rb'))

X_test = df[df['split'] == 'test'][frozen['features']]
y_test = df[df['split'] == 'test']['quality_label']

# Dự đoán
y_prob_test = models['candidate'].predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= frozen['chosen_threshold']).astype(int)

# Đánh giá
cm = confusion_matrix(y_test, y_pred_test)
flag_rate = y_pred_test.sum() / len(y_test)

print(f"Sử dụng Threshold đóng băng: {frozen['chosen_threshold']:.2f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
print(f"Test Flag Rate: {flag_rate:.4f} (Ràng buộc < 0.20: {'Thỏa mãn' if flag_rate <= 0.20 else 'Không thỏa mãn'})")

# Giả sử Cost per 100 = (FP * cost_FP + FN * cost_FN) / (N/100). Đề bài yêu cầu tính C_100.
# Ở đây ta giả định FP cost = 1, FN cost = 5 (theo thông lệ y tế)
TN, FP, FN, TP = cm.ravel()
cost_100 = (FP * 1 + FN * 5) / (len(y_test) / 100)
print(f"Cost per 100 records (Giả định chi phí FP=1, FN=5): {cost_100:.2f}")

# Lưu kết quả test để phân tích ở bài 7
df_test = df[df['split'] == 'test'].copy()
df_test['y_pred'] = y_pred_test
df_test['y_prob'] = y_prob_test
df_test.to_csv('test_results.csv', index=False)