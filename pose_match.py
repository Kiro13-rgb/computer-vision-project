import os
import cv2
import numpy as np
import argparse

def compute_hu_moments(image_path):
    """
    Загружает изображение по указанному пути, преобразует его в бинарное,
    находит основной контур (самый большой) и вычисляет логарифмированые Hu-моменты.
    Возвращает массив из 7 значений Hu-моментов.
    """
    # Чтение изображения в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # Применяем бинаризацию (через Otsu) для выделения силуэта
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"Контуры не найдены на изображении: {image_path}")

    # Берём самый большой контур (предполагаем, что это фигура человека)
    largest_contour = max(contours, key=cv2.contourArea)

    # Вычисляем моменты контура и Hu-моменты
    moments = cv2.moments(largest_contour)
    hu = cv2.HuMoments(moments).flatten()

    # Логарифмируем Hu-моменты для устойчивости (аналог -sign * log10(abs(x)))
    for i in range(len(hu)):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)

    return hu

def match_templates(test_image_path, templates_dir):
    """
    Считает Hu-моменты для тестового изображения
    и для каждого файла-шаблона в папке templates_dir.
    Возвращает список кортежей (имя_файла, расстояние),
    отсортированный по возрастанию расстояния (L2-норма).
    """
    test_hu = compute_hu_moments(test_image_path)
    results = []

    # Проходим по всем файлам в директории шаблонов
    for filename in os.listdir(templates_dir):
        file_path = os.path.join(templates_dir, filename)
        # Проверяем расширение
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            tpl_hu = compute_hu_moments(file_path)
            dist = np.linalg.norm(test_hu - tpl_hu)
            results.append((filename, dist))
        except Exception as e:
            # Игнорируем файлы, где не получилось вычислить моменты
            print(f"[WARNING] Не удалось обработать шаблон {filename}: {e}")

    # Сортируем по расстоянию (меньше = больше похож)
    results.sort(key=lambda x: x[1])
    return results

def main():
    parser = argparse.ArgumentParser(description="Распознавание поз по силуэту (шаблонный метод).")
    parser.add_argument("test_image", help="Путь к тестовому изображению силуэта (PNG/JPEG/BMP).")
    parser.add_argument("templates_dir", help="Путь к папке с шаблонами поз (силуэтами).")
    args = parser.parse_args()

    test_img = args.test_image
    tpl_dir = args.templates_dir

    if not os.path.isfile(test_img):
        print(f"Ошибка: файл тестового изображения не найден: {test_img}")
        return
    if not os.path.isdir(tpl_dir):
        print(f"Ошибка: папка с шаблонами не найдена: {tpl_dir}")
        return

    print(f"Распознаём позу для изображения: {test_img}")
    print(f"Ищем в папке шаблонов: {tpl_dir}\n")

    matches = match_templates(test_img, tpl_dir)

    if not matches:
        print("Не найдено ни одного подходящего шаблона.")
        return

    print("Топ-5 наиболее похожих поз (шаблон, расстояние):")
    for i, (name, dist) in enumerate(matches[:5], start=1):
        print(f"{i}. {name} -> {dist:.6f}")

if __name__ == "__main__":
    main()
