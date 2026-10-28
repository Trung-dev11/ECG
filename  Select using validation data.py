 import pandas as pd
import json
import pickle
import numpy as np

print("=== TASK 4: SELECT USING VALIDATION DATA ===")

df = pd.read_json('ecg_quality_features.jsonl', lines=True)
df['split'] = df['participant_id'].map(json.load(open('participant_split.json')))
features = json.load(open('selected_features.json'))['features']
models = pickle.load(open('models.pkl', 'rb'))
candidate = models['candidate']

# Lấy tập Validation
val_df = df[df['split'] == 'validation']
X_val = val_df[features]
y_val = val_df['quality_label']

# Dự đoán xác suất
y_prob_val = candidate.predict_proba(X_val)[:, 1]

# Khảo sát các threshold
thresholds = np.arange(0.1, 1.0, 0.05)
best_threshold = 0.5 # Mặc định
print(f"{'Threshold':<10} | {'Flag Rate':<15} | {'Valid? (<= 0.2)'}")

for t in thresholds:
    y_pred = (y_prob_val >= t).astype(int)
    # Flag rate = tỷ lệ dự đoán positive (gắn cờ kiểm tra)
    flag_rate = y_pred.sum() / len(y_pred) 
    is_valid = flag_rate <= 0.20
    print(f"{t:<10.2f} | {flag_rate:<15.4f} | {is_valid}")
    
    # Logic chọn rule: Chọn threshold nhỏ nhất (để bắt được nhiều lỗi nhất) nhưng flag_rate <= 0.20
    if is_valid and best_threshold == 0.5 and t != 0.5:
        # Nếu ngưỡng 0.5 không đạt, tìm ngưỡng gần nhất đạt
        pass

# Giả sử qua bảng, ta chọn ngưỡng tốt nhất thỏa mãn (ví dụ 0.40)
# Trong bài tự động hóa, ta lập trình để lấy:
valid_thresholds = [t for t in thresholds if (y_prob_val >= t).astype(int).sum() / len(y_val) <= 0.20]
chosen_threshold = min(valid_thresholds) if valid_thresholds else 0.5

print(f"\n=> Quyết định chọn Threshold: {chosen_threshold:.2f} (Đảm bảo Flag rate <= 0.20)")

# Lưu threshold để đóng băng (Freeze)
with open('optimal_threshold.txt', 'w') as f:
    f.write(str(chosen_threshold))