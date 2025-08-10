import os
import torch
from ultralytics import YOLO
import yaml
from tqdm import tqdm
import time


def main():
    print("🚀 === ОБУЧЕНИЕ YOLO МОДЕЛИ НА СЕРВЕРЕ ===")
    print("=" * 50)

    # Проверка CUDA
    print("🔍 Проверка оборудования...")
    if torch.cuda.is_available():
        print(f"✅ CUDA доступен: {torch.cuda.get_device_name(0)}")
        print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("⚠️  CUDA недоступен, используется CPU")

    # Базовая директория на сервере
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    YAML_PATH = os.path.join(DATA_DIR, 'data.yaml')

    print(f"\n📁 Базовая директория: {BASE_DIR}")
    print(f"📁 Директория данных: {DATA_DIR}")
    print(f"📄 Файл конфигурации: {YAML_PATH}")

    # Проверяем существование файлов
    print("\n🔍 Проверка файлов...")
    with tqdm(total=4, desc="Проверка файлов", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

        if not os.path.exists(YAML_PATH):
            print(f"❌ Ошибка: файл {YAML_PATH} не найден!")
            print("💡 Сначала запустите prepare_server_dataset.py")
            return
        pbar.update(1)

        # Загружаем YAML для проверки
        with open(YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)

        train_path = os.path.join(DATA_DIR, data_config['train'])
        val_path = os.path.join(DATA_DIR, data_config['val'])

        if not os.path.exists(val_path) or len(os.listdir(val_path)) == 0:
            print("⚠️ Валидационные данные отсутствуют, используем тренировочные для валидации")
            val_path = train_path
            val_labels_path = train_labels_path  # Обновляем путь к меткам

        if not os.path.exists(train_path):
            print(f"❌ Ошибка: папка с обучающими данными не найдена: {train_path}")
            return
        pbar.update(1)

        if not os.path.exists(val_path):
            print(f"❌ Ошибка: папка с валидационными данными не найдена: {val_path}")
            return
        pbar.update(1)

        # Проверяем наличие аннотаций
        train_labels_path = os.path.join(DATA_DIR, 'train', 'labels')
        val_labels_path = os.path.join(DATA_DIR, 'val', 'labels')

        if not os.path.exists(train_labels_path):
            print(f"❌ Ошибка: папка с обучающими аннотациями не найдена: {train_labels_path}")
            return
        pbar.update(1)

    print("✅ Все файлы найдены!")

    # Подсчитываем количество файлов
    print("\n📊 Подсчет данных...")
    with tqdm(total=4, desc="Подсчет файлов", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

        train_images = len([f for f in os.listdir(train_path) if f.endswith('.jpg')])
        pbar.set_postfix({"train_img": train_images})
        pbar.update(1)

        val_images = len([f for f in os.listdir(val_path) if f.endswith('.jpg')])
        pbar.set_postfix({"val_img": val_images})
        pbar.update(1)

        train_labels = len([f for f in os.listdir(train_labels_path) if f.endswith('.txt')])
        pbar.set_postfix({"train_labels": train_labels})
        pbar.update(1)

        val_labels = len([f for f in os.listdir(val_labels_path) if f.endswith('.txt')])
        pbar.set_postfix({"val_labels": val_labels})
        pbar.update(1)

    print(f"📈 Найдено {train_images} обучающих изображений")
    print(f"📈 Найдено {val_images} валидационных изображений")
    print(f"📈 Найдено {train_labels} обучающих аннотаций")
    print(f"📈 Найдено {val_labels} валидационных аннотаций")

    if train_images == 0:
        print("❌ Ошибка: нет обучающих изображений!")
        print("💡 Сначала запустите prepare_server_dataset.py")
        return

    if train_labels == 0:
        print("❌ Ошибка: нет обучающих аннотаций!")
        return

    # Инициализируем модель
    print("\n🤖 Загрузка модели YOLOv8...")
    with tqdm(total=1, desc="Загрузка модели", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        model = YOLO('yolov8n.pt')  # Используем базовую модель
        pbar.update(1)

    # Создаем папку для результатов
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Обучаем модель
    print("\n🎯 Начинаем обучение...")
    print("=" * 50)

    start_time = time.time()

    try:
        results = model.train(
            data=YAML_PATH,
            epochs=50,
            imgsz=640,
            batch=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            patience=10,  # Остановка если нет улучшений
            save=True,
            project=results_dir,
            name="yolo_training",
            verbose=True
        )

        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 50)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 50)
        print(f"⏱️  Время обучения: {training_time / 3600:.2f} часов ({training_time / 60:.1f} минут)")
        print(f"💾 Лучшая модель сохранена в: {results.save_dir}")

        # Выводим финальные метрики
        if hasattr(results, 'results_dict'):
            print("\n📊 Финальные метрики:")
            print("-" * 30)
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    if 'mAP' in key:
                        print(f"🎯 {key}: {value:.4f}")
                    elif 'precision' in key:
                        print(f"🎯 {key}: {value:.4f}")
                    elif 'recall' in key:
                        print(f"🎯 {key}: {value:.4f}")
                    else:
                        print(f"📈 {key}: {value:.4f}")

        print(f"\n🎯 Модель готова к использованию!")
        print(f"📁 Путь к лучшей модели: {results.save_dir}/weights/best.pt")

    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        return


if __name__ == "__main__":
    main()