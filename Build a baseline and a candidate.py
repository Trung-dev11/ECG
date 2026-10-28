   import pandas as pd
import json
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

print("=== TASK 3: BUILD BASELINE AND CANDIDATE ===")

# Load data và cấu hình
df = pd.read_json('ecg_quality_features.jsonl', lines=True)
df['split'] = df['participant_id'].map(json.load(open('participant_split.json')))
config = json.load(open('selected_features.json'))

features = config['features']
target = config['target']

# Lấy tập Train
train_df = df[df['split'] == 'train']
X_train = train_df[features]
y_train = train_df[target]

# 1. Baseline: Dự đoán dựa trên tỷ lệ lớp (Class-prior)
baseline_model = DummyClassifier(strategy='prior')
baseline_model.fit(X_train, y_train)

# 2. Candidate: Shallow Decision Tree (Cây quyết định nông - dễ giải thích)
# Giới hạn max_depth=3 để tránh overfitting và giữ tính minh bạch
candidate_model = DecisionTreeClassifier(max_depth=3, random_state=42)
candidate_model.fit(X_train, y_train)

print("Đã huấn luyện Baseline (Dummy) và Candidate (Decision Tree d=3) trên tập Train.")

# Lưu mô hình
with open('models.pkl', 'wb') as f:
    pickle.dump({'baseline': baseline_model, 'candidate': candidate_model}, f)
print("Đã lưu các mô hình vào 'models.pkl'")