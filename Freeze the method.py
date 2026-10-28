  import json

print("=== TASK 5: FREEZE THE METHOD ===")

with open('selected_features.json', 'r') as f:
    config = json.load(f)

with open('optimal_threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

frozen_config = {
    "features": config['features'],
    "model_type": "DecisionTreeClassifier",
    "hyperparameters": {"max_depth": 3, "random_state": 42},
    "preprocessing": "None required for Tree algorithms",
    "chosen_threshold": threshold
}

with open('frozen_method.json', 'w') as f:
    json.dump(frozen_config, f, indent=4)

print("Đã đóng băng quy trình (Frozen Method). Không được phép tinh chỉnh lại sau bước này!")
print(json.dumps(frozen_config, indent=2))