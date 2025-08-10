import os
import shutil
from PIL import Image
from tqdm import tqdm


def create_server_dataset():
    """
    Создает правильную структуру данных для YOLO на сервере
    """
    print("🚀 === ПОДГОТОВКА ДАННЫХ ДЛЯ YOLO ===")
    print("=" * 50)

    # Базовая директория на сервере
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')

    # Пути к данным
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')
    TRAIN_LABELS_DIR = os.path.join(DATA_DIR, 'train', 'labels')
    VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'val', 'images')
    VAL_LABELS_DIR = os.path.join(DATA_DIR, 'val', 'labels')

    print(f"📁 Базовая директория: {BASE_DIR}")
    print(f"📁 Директория данных: {DATA_DIR}")

    # Создаем структуру папок
    print("\n📂 Создание структуры папок...")
    with tqdm(total=4, desc="Создание папок", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
        pbar.update(1)
        os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
        pbar.update(1)
        os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
        pbar.update(1)
        os.makedirs(VAL_LABELS_DIR, exist_ok=True)
        pbar.update(1)

    # Пути к исходным данным (предполагаем, что они скопированы в BASE_DIR)
    source_sequences = os.path.join(BASE_DIR, 'dataset', 'sequences')
    source_labels = os.path.join(BASE_DIR, 'dataset', 'labels')

    # Проверяем существование исходных данных
    if not os.path.exists(source_sequences):
        print(f"❌ Ошибка: исходные данные не найдены в {source_sequences}")
        print("💡 Убедитесь, что папка 'dataset' скопирована в BASE_DIR")
        return

    # Копируем данные из train
    train_sequences = os.path.join(source_sequences, 'train')
    train_labels = os.path.join(source_labels, 'train')

    train_count = 0
    if os.path.exists(train_sequences):
        print(f"\n📁 Обработка обучающих данных...")
        video_folders = [f for f in os.listdir(train_sequences) if os.path.isdir(os.path.join(train_sequences, f))]

        with tqdm(total=len(video_folders), desc="Обработка train",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for video_folder in video_folders:
                video_path = os.path.join(train_sequences, video_folder)

                # Копируем изображения
                img_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                for img_file in img_files:
                    src_img = os.path.join(video_path, img_file)
                    dst_img = os.path.join(TRAIN_IMAGES_DIR, f"{video_folder}_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    train_count += 1

                # Копируем аннотации
                video_label_path = os.path.join(train_labels, video_folder)
                if os.path.exists(video_label_path):
                    label_files = [f for f in os.listdir(video_label_path) if f.endswith('.txt')]
                    for label_file in label_files:
                        src_label = os.path.join(video_label_path, label_file)
                        dst_label = os.path.join(TRAIN_LABELS_DIR, f"{video_folder}_{label_file}")
                        shutil.copy2(src_label, dst_label)

                pbar.set_postfix({"images": train_count})
                pbar.update(1)

    # Копируем данные из val
    val_sequences = os.path.join(source_sequences, 'val')
    val_labels = os.path.join(source_labels, 'val')

    val_count = 0
    if os.path.exists(val_sequences):
        print(f"\n📁 Обработка валидационных данных...")
        video_folders = [f for f in os.listdir(val_sequences) if os.path.isdir(os.path.join(val_sequences, f))]

        with tqdm(total=len(video_folders), desc="Обработка val",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for video_folder in video_folders:
                video_path = os.path.join(val_sequences, video_folder)

                # Копируем изображения
                img_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                for img_file in img_files:
                    src_img = os.path.join(video_path, img_file)
                    dst_img = os.path.join(VAL_IMAGES_DIR, f"{video_folder}_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    val_count += 1

                # Копируем аннотации
                video_label_path = os.path.join(val_labels, video_folder)
                if os.path.exists(video_label_path):
                    label_files = [f for f in os.listdir(video_label_path) if f.endswith('.txt')]
                    for label_file in label_files:
                        src_label = os.path.join(video_label_path, label_file)
                        dst_label = os.path.join(VAL_LABELS_DIR, f"{video_folder}_{label_file}")
                        shutil.copy2(src_label, dst_label)

                pbar.set_postfix({"images": val_count})
                pbar.update(1)

    print(f"\n📊 Результаты подготовки:")
    print(f"✅ Создано {train_count} обучающих изображений")
    print(f"✅ Создано {val_count} валидационных изображений")
    print(f"📁 Структура данных готова в: {DATA_DIR}")

    # Создаем data.yaml для сервера
    print(f"\n📄 Создание конфигурационного файла...")
    yaml_content = f"""path: {DATA_DIR}
train: train/images
val: val/images

nc: 10
names: ["pedestrian", "person", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
"""

    yaml_path = os.path.join(DATA_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Создан файл конфигурации: {yaml_path}")
    print(f"\n🎉 Подготовка данных завершена успешно!")
    print(f"🚀 Теперь можно запускать обучение: python train_server.py")


if __name__ == "__main__":
    create_server_dataset()
