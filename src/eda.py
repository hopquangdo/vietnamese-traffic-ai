import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Đường dẫn thư mục chứa ảnh đã xử lý
image_dir = "../dataset/images/processed_images"
label_dir = "../dataset/labels"

# Regex để tìm class_id từ tên file
pattern = re.compile(r'cls(\d+)\.jpg$')

class_ids = []

# Lặp qua toàn bộ file ảnh
for fname in os.listdir(image_dir):
    match = pattern.search(fname)
    if match:
        class_id = int(match.group(1))
        class_ids.append(class_id)

# Đếm số lượng ảnh mỗi class
counter = Counter(class_ids)

# In thông tin
print(f"📌 Tổng ảnh: {len(class_ids)}")
print(f"📌 Số lớp: {len(counter)}")
print(f"📌 Số ảnh / lớp:")
for class_id, count in sorted(counter.items()):
    print(f" - Class {class_id:2d}: {count} ảnh")

# Vẽ biểu đồ phân bố class
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))
sns.barplot(x=list(counter.keys()), y=list(counter.values()), palette="Blues_d")
plt.title("Phân bố số ảnh theo class")
plt.xlabel("Class ID")
plt.ylabel("Số ảnh")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
