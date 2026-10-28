 import pandas as pd
import json

print("=== TASK 2: AUDIT FEATURE ELIGIBILITY ===")

# Đọc dữ liệu theo hướng dẫn trong README.md
df = pd.read_json('ecg_quality_features.jsonl', lines=True)
with open('participant_split.json', 'r') as f:
    split_map = json.load(f)
df['split'] = df['participant_id'].map(split_map)

# Phân loại các cột
all_columns = df.columns.tolist()
target = 'quality_label'
identifiers = ['segment_id', 'participant_id', 'device_id'] # device_id dùng để phân tích sau, không đưa vào model
post_review = ['review_note']
constants = [col for col in df.columns if df[col].nunique() <= 1] # duration_s luôn = 10

# Lựa chọn đặc trưng hợp lệ (có sẵn tại thời điểm dự đoán)
eligible_features = [c for c in all_columns if c not in identifiers + post_review + constants + [target, 'split']]

print(f"- Target (Không dùng làm input): {target}")
print(f"- Identifiers (Định danh, loại bỏ): {identifiers}")
print(f"- Post-review (Thông tin sau đánh giá, rò rỉ dữ liệu, loại bỏ): {post_review}")
print(f"- Constant fields (Hằng số, không có ý nghĩa phân loại, loại bỏ): {constants}")
print(f"=> Eligible Predictors (Đặc trưng được giữ lại): {eligible_features}")

# Lưu danh sách features để các bài sau dùng
with open('selected_features.json', 'w') as f:
    json.dump({'features': eligible_features, 'target': target}, f)
print("Đã lưu cấu hình features vào 'selected_features.json'")