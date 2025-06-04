# generate_masks_from_bbox.py

import os
import json
import cv2
import numpy as np

def extract_bbox_masks_from_coco(data_root: str, split: str = "train"):
    # Путь к JSON-аннотациям, например:
    # C:\Users\Amir\Pose-Integrated-Firearm-Detection-Dataset-4\train.json
    ann_file   = os.path.join(data_root, f"{split}.json")
    # Папка с картинками, например:
    # C:\Users\Amir\Pose-Integrated-Firearm-Detection-Dataset-4\train
    images_dir = os.path.join(data_root, split)
    # Папка, куда будем записывать прямоугольные маски:
    # C:\Users\Amir\Pose-Integrated-Firearm-Detection-Dataset-4\train\masks
    masks_out  = os.path.join(images_dir, "masks")

    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO JSON не найден: {ann_file}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Папка {split} не найдена: {images_dir}")

    os.makedirs(masks_out, exist_ok=True)

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Собираем словарь image_id -> file_name
    id2file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    # Собираем bbox для каждого image_id
    bbox_per_image = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        bbox   = ann.get("bbox")
        if img_id is None or bbox is None:
            continue
        bbox_per_image.setdefault(img_id, []).append(bbox)

    for img_id, bboxes in bbox_per_image.items():
        filename = id2file.get(img_id)
        if not filename:
            continue

        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Не удалось открыть изображение: {img_path}")
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for bbox in bboxes:
            # bbox = [x, y, width, height]
            x, y, bw, bh = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(min(x + bw, w - 1))
            y2 = int(min(y + bh, h - 1))
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(masks_out, f"{base_name}.png")
        cv2.imwrite(out_path, mask)

    print(f"Маски для split='{split}' сгенерированы в: {masks_out}")


if __name__ == "__main__":
    # Укажите реальный путь к распакованному датасету
    DATA_ROOT = r"C:\Users\Amir\Pose-Integrated-Firearm-Detection-Dataset-4"


    for sp in ["train", "valid", "test"]:
        extract_bbox_masks_from_coco(DATA_ROOT, split=sp)
