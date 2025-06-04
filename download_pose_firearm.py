# download_pose_firearm.py
from roboflow import Roboflow
import sys

def download_pose_firearm_dataset():
    # -------------------------------
    # ↓ Вставьте сюда свой Download API Key (скопированный из Roboflow → Download dataset)
    DOWNLOAD_API_KEY = "9Pw3h1GpSV0SOEAmFtt2"
    # -------------------------------

    WORKSPACE   = "weapon-detection-oss9n"
    PROJECT     = "pose-integrated-firearm-detection-dataset"
    VERSION_NUM = 4   # в вашем сниппете указана версия 4

    # Проверка, что ключ не пустой
    if DOWNLOAD_API_KEY.startswith("ВАШ_") or DOWNLOAD_API_KEY.strip() == "":
        print("❗ Ошибка: не задан Download API Key.")
        print("   Скопируйте ключ из окна «Download dataset» (Roboflow → Dataset → Download).")
        sys.exit(1)

    print("🔑 Используем Download API Key:", DOWNLOAD_API_KEY)
    rf = Roboflow(api_key=DOWNLOAD_API_KEY)

    try:
        project = rf.workspace(WORKSPACE).project(PROJECT)
    except Exception as e:
        print(f"❗ Не удалось получить проект '{WORKSPACE}/{PROJECT}': {e}")
        sys.exit(1)

    print(f"🚀 Скачиваем версию {VERSION_NUM} датасета '{PROJECT}' в формате COCO…")
    try:
        version = project.version(VERSION_NUM)
        dataset = version.download("coco")
    except Exception as e:
        print(f"❗ Ошибка при скачивании датасета: {e}")
        sys.exit(1)

    data_path = dataset.location
    print("✅ Датасет успешно скачан и распакован в:")
    print("   ", data_path)
    return data_path

if __name__ == "__main__":
    download_pose_firearm_dataset()
