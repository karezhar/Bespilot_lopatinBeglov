import os
import glob


def check_dataset():
    """
    Проверяет структуру данных на сервере
    """
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')

    print("=== Проверка структуры данных на сервере ===")
    print(f"Базовая директория: {BASE_DIR}")

    # Проверяем основные папки
    required_dirs = [
        'VisDrone2019-VID-train',
        'dataset',
        'data'
    ]

    for dir_name in required_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}: найдена")
        else:
            print(f"✗ {dir_name}: НЕ НАЙДЕНА")

    print("\n=== Проверка исходных данных ===")

    # Проверяем VisDrone данные
    visdrone_ann = os.path.join(BASE_DIR, 'VisDrone2019-VID-train', 'annotations')
    visdrone_seq = os.path.join(BASE_DIR, 'VisDrone2019-VID-train', 'sequences')

    if os.path.exists(visdrone_ann):
        ann_files = len([f for f in os.listdir(visdrone_ann) if f.endswith('.txt')])
        print(f"✓ Аннотации VisDrone: {ann_files} файлов")
    else:
        print("✗ Аннотации VisDrone: НЕ НАЙДЕНЫ")

    if os.path.exists(visdrone_seq):
        seq_dirs = len([d for d in os.listdir(visdrone_seq) if os.path.isdir(os.path.join(visdrone_seq, d))])
        print(f"✓ Последовательности VisDrone: {seq_dirs} папок")
    else:
        print("✗ Последовательности VisDrone: НЕ НАЙДЕНЫ")

    print("\n=== Проверка конвертированных данных ===")

    # Проверяем dataset
    dataset_labels = os.path.join(BASE_DIR, 'dataset', 'labels')
    dataset_seq = os.path.join(BASE_DIR, 'dataset', 'sequences')

    if os.path.exists(dataset_labels):
        label_dirs = len([d for d in os.listdir(dataset_labels) if os.path.isdir(os.path.join(dataset_labels, d))])
        print(f"✓ Конвертированные аннотации: {label_dirs} папок")
    else:
        print("✗ Конвертированные аннотации: НЕ НАЙДЕНЫ")

    if os.path.exists(dataset_seq):
        seq_dirs = len([d for d in os.listdir(dataset_seq) if os.path.isdir(os.path.join(dataset_seq, d))])
        print(f"✓ Конвертированные изображения: {seq_dirs} папок")
    else:
        print("✗ Конвертированные изображения: НЕ НАЙДЕНЫ")

    print("\n=== Проверка подготовленных данных ===")

    # Проверяем data
    data_train_img = os.path.join(BASE_DIR, 'data', 'train', 'images')
    data_train_label = os.path.join(BASE_DIR, 'data', 'train', 'labels')
    data_val_img = os.path.join(BASE_DIR, 'data', 'val', 'images')
    data_val_label = os.path.join(BASE_DIR, 'data', 'val', 'labels')
    data_yaml = os.path.join(BASE_DIR, 'data', 'data.yaml')

    if os.path.exists(data_train_img):
        train_images = len([f for f in os.listdir(data_train_img) if f.endswith('.jpg')])
        print(f"✓ Обучающие изображения: {train_images} файлов")
    else:
        print("✗ Обучающие изображения: НЕ НАЙДЕНЫ")

    if os.path.exists(data_train_label):
        train_labels = len([f for f in os.listdir(data_train_label) if f.endswith('.txt')])
        print(f"✓ Обучающие аннотации: {train_labels} файлов")
    else:
        print("✗ Обучающие аннотации: НЕ НАЙДЕНЫ")

    if os.path.exists(data_val_img):
        val_images = len([f for f in os.listdir(data_val_img) if f.endswith('.jpg')])
        print(f"✓ Валидационные изображения: {val_images} файлов")
    else:
        print("✗ Валидационные изображения: НЕ НАЙДЕНЫ")

    if os.path.exists(data_val_label):
        val_labels = len([f for f in os.listdir(data_val_label) if f.endswith('.txt')])
        print(f"✓ Валидационные аннотации: {val_labels} файлов")
    else:
        print("✗ Валидационные аннотации: НЕ НАЙДЕНЫ")

    if os.path.exists(data_yaml):
        print("✓ data.yaml: найден")
    else:
        print("✗ data.yaml: НЕ НАЙДЕН")

    print("\n=== Рекомендации ===")

    if not os.path.exists(visdrone_ann) or not os.path.exists(visdrone_seq):
        print("1. Скопируйте папку 'VisDrone2019-VID-train' в ~/Bespilot_lopatinBeglov/")

    if not os.path.exists(dataset_labels):
        print("2. Запустите: python convert_annotations_server.py")

    if not os.path.exists(data_train_img):
        print("3. Запустите: python prepare_server_dataset.py")

    if not os.path.exists(data_yaml):
        print("4. Запустите: python prepare_server_dataset.py")

    if all([os.path.exists(data_train_img), os.path.exists(data_yaml), os.path.exists(data_train_label)]):
        print("✓ Все готово для обучения! Запустите: python train_server.py")


if __name__ == "__main__":
    check_dataset()