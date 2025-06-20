import os
import cv2
import numpy as np


def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def process_raw_data(
        input,
        save_path,
        img_size=(32, 32)
):
    """
    Đọc ảnh gốc và nhãn YOLO, crop các object, resize về img_size,
    và lưu toàn bộ vào 1 file .npz

    Args:
        input (str): thư mục chứa dữ liệu
        save_path (str): đường dẫn lưu file .npz
        img_size (tuple): kích thước resize ảnh object
    """
    input_images = f"{input}/images"
    input_labels = f"{input}/labels"
    X, y = [], []

    for filename in os.listdir(input_images):
        if not filename.endswith('.jpg'):
            continue

        image_path = os.path.join(input_images, filename)
        label_path = os.path.join(input_labels, filename.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])
                x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, box_w, box_h, w, h)

                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                cropped_resized = cv2.resize(cropped, img_size)
                X.append(cropped_resized)
                y.append(class_id)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, y=y)
    print(f"[✓] Đã lưu {len(X)} ảnh object vào {save_path}")


# Ví dụ sử dụng:
# process_raw_data(
#     input="../../../dataset/raw/valid",
#     save_path="../../../dataset/processed/valid/dataset.npz",
#     img_size=(32, 32)
# )
