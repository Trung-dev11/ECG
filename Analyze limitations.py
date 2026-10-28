  import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("=== TASK 7: ANALYZE LIMITATIONS ===")

df_test = pd.read_csv('test_results.csv')

# 1. So sánh hiệu suất theo Device
print("Hiệu suất (Số lượng & Flag Rate) phân rã theo thiết bị:")
device_stats = df_test.groupby('device_id').agg(
    Total_Segments=('segment_id', 'count'),
    Flagged=('y_pred', 'sum'),
    Actual_Positives=('quality_label', 'sum')
)
device_stats['Flag_Rate'] = device_stats['Flagged'] / device_stats['Total_Segments']
print(device_stats)

print("\nGiải thích: Mặc dù hiệu suất trên D1 và D2 trong tập test có thể tốt, nhưng điều này không đảm bảo model sẽ hoạt động tương tự trên một thiết bị mới (D3) vì model chưa được học các đặc trưng/nhiễu cụ thể của thiết bị mới.")

# 2. Vẽ Confusion Matrix chung
plt.figure(figsize=(6,4))
cm = confusion_matrix(df_test['quality_label'], df_test['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("Đã lưu biểu đồ 'confusion_matrix.png'")