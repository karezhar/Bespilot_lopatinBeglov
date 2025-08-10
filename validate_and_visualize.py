import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import shutil


def main():
    print("🚀 === ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ НА ВАЛИДАЦИОННЫХ ДАННЫХ ===")
    print("=" * 60)

    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Визуализация результатов детекции')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Название последовательности (например uav0000086_00000_v)')
    parser.add_argument('--model', type=str, default='best.pt',
                        help='Путь к модели YOLO')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Порог уверенности для детекции')
    parser.add_argument('--fps', type=int, default=30,
                        help='Частота кадров для видео')
    args = parser.parse_args()

    # Базовые пути
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')
    VAL_DIR = os.path.join(BASE_DIR, 'VisDrone2019-VID-val')
    RESULTS_DIR = os.path.join(BASE_DIR, 'validation_results')

    # Создаем рабочую директорию
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Проверка CUDA
    print("\n🔍 Проверка оборудования...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device.upper()}")

    # Загрузка модели
    print("\n🤖 Загрузка модели YOLO...")
    model_path = os.path.join(BASE_DIR, 'results', 'yolo_training2', 'weights', args.model)

    if not os.path.exists(model_path):
        # Попробуем альтернативный путь
        model_path = os.path.join(BASE_DIR, 'results', 'yolo_training', args.model)
        if not os.path.exists(model_path):
            print(f"❌ Ошибка: модель не найдена: {args.model}")
            return

    model = YOLO(model_path)
    print(f"✅ Модель загружена: {os.path.basename(model_path)}")

    # Пути к данным
    seq_path = os.path.join(VAL_DIR, 'sequences', args.sequence)
    ann_path = os.path.join(VAL_DIR, 'annotations', f"{args.sequence}.txt")

    if not os.path.exists(seq_path):
        print(f"❌ Ошибка: последовательность не найдена: {seq_path}")
        return

    # Получаем список изображений
    images = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])
    if not images:
        print(f"❌ Ошибка: нет изображений в последовательности: {seq_path}")
        return

    print(f"\n📹 Обработка последовательности: {args.sequence}")
    print(f"  Количество кадров: {len(images)}")
    print(f"  FPS: {args.fps}")
    print(f"  Порог уверенности: {args.conf}")

    # Определяем размер видео по первому изображению
    first_image = cv2.imread(os.path.join(seq_path, images[0]))
    height, width, _ = first_image.shape

    # Создаем VideoWriter
    output_video = os.path.join(RESULTS_DIR, f"{args.sequence}_detection.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, args.fps, (width, height))

    # Обработка кадров с прогресс-баром
    print("\n🎬 Генерация видео с детекцией...")
    progress_bar = tqdm(images, desc="Обработка кадров", unit="кадр")

    for img_name in progress_bar:
        img_path = os.path.join(seq_path, img_name)
        frame = cv2.imread(img_path)

        # Выполняем детекцию
        results = model.predict(
            source=frame,
            conf=args.conf,
            device=device,
            verbose=False
        )

        # Получаем результаты детекции
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Визуализируем результаты с зеленой окантовкой
        for box, cls_id, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)

            # Рисуем зеленый прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Подпись с классом и уверенностью
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Добавляем кадр в видео
        out.write(frame)

    out.release()
    progress_bar.close()

    # Создаем видео с оригинальными кадрами для сравнения
    print("\n🎬 Создание видео с оригинальными кадрами...")
    original_video = os.path.join(RESULTS_DIR, f"{args.sequence}_original.mp4")
    orig_out = cv2.VideoWriter(original_video, fourcc, args.fps, (width, height))

    for img_name in tqdm(images, desc="Обработка оригиналов", unit="кадр"):
        img_path = os.path.join(seq_path, img_name)
        frame = cv2.imread(img_path)
        orig_out.write(frame)

    orig_out.release()

    # Создаем комбинированное видео (side-by-side)
    print("\n🔄 Создание комбинированного видео...")
    combined_video = os.path.join(RESULTS_DIR, f"{args.sequence}_comparison.mp4")

    # Открываем оба видео
    cap_orig = cv2.VideoCapture(original_video)
    cap_det = cv2.VideoCapture(output_video)

    # Создаем новое видео
    combined_width = width * 2
    combined_out = cv2.VideoWriter(combined_video, fourcc, args.fps, (combined_width, height))

    frame_count = len(images)
    for _ in tqdm(range(frame_count), desc="Комбинирование", unit="кадр"):
        ret_orig, frame_orig = cap_orig.read()
        ret_det, frame_det = cap_det.read()

        if not ret_orig or not ret_det:
            break

        # Объединяем кадры горизонтально
        combined_frame = np.hstack((frame_orig, frame_det))

        # Добавляем подписи
        cv2.putText(combined_frame, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined_frame, "Detection", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined_out.write(combined_frame)

    # Закрываем все ресурсы
    cap_orig.release()
    cap_det.release()
    combined_out.release()

    # Удаляем промежуточные файлы
    os.remove(original_video)
    os.remove(output_video)

    print("\n" + "=" * 60)
    print("🎉 ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)
    print(f"💾 Видео сохранено в: {combined_video}")
    print(f"\n📥 Для скачивания файла используйте команду:")
    print(f"scp user001@server_ip:{combined_video} .")
    print(f"\n🖼️ Размер видео: {width * 2}x{height} (оригинал + детекция)")
    print(f"⏱️ Продолжительность: {frame_count / args.fps:.1f} секунд")


if __name__ == "__main__":
    main()