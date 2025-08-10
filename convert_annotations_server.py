import os
import shutil
from PIL import Image


def convert_annotations_server():
    """
    Конвертирует аннотации из формата VisDrone в формат YOLO на сервере
    и копирует изображения в структурированную папку dataset
    """
    # Базовая директория на сервере
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')

    # Создаем структуру папок для YOLO
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

    # Обрабатываем оба набора данных: train и val
    for data_type in ['train', 'val']:
        print(f"\n{'=' * 50}")
        print(f"Обработка {data_type.upper()} данных")
        print(f"{'=' * 50}")

        SEQ_DIR = os.path.join(DATASET_DIR, 'sequences', data_type)
        LABEL_DIR = os.path.join(DATASET_DIR, 'labels', data_type)

        # Пути к исходным данным VisDrone
        ann_dir = os.path.join(BASE_DIR, f'VisDrone2019-VID-{data_type}', 'annotations')
        seq_dir = os.path.join(BASE_DIR, f'VisDrone2019-VID-{data_type}', 'sequences')

        print(f"Директория аннотаций: {ann_dir}")
        print(f"Директория изображений: {seq_dir}")
        print(f"Директория для меток: {LABEL_DIR}")
        print(f"Директория для изображений: {SEQ_DIR}")

        # Создаем структуру папок
        os.makedirs(SEQ_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)

        # Проверяем существование исходных данных
        if not os.path.exists(ann_dir):
            print(f"Предупреждение: директория аннотаций не найдена: {ann_dir}")
            continue

        if not os.path.exists(seq_dir):
            print(f"Предупреждение: директория изображений не найдена: {seq_dir}")
            continue

        converted_count = 0
        skipped_count = 0

        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith('.txt'):
                continue

            video_id = os.path.splitext(ann_file)[0]
            ann_path = os.path.join(ann_dir, ann_file)
            src_video_img_dir = os.path.join(seq_dir, video_id)
            dst_video_img_dir = os.path.join(SEQ_DIR, video_id)

            # Копируем папку с изображениями
            if not os.path.exists(src_video_img_dir):
                print(f'Пропущено: нет папки изображений {src_video_img_dir}')
                skipped_count += 1
                continue

            # Копируем только если папка еще не существует
            if not os.path.exists(dst_video_img_dir):
                shutil.copytree(src_video_img_dir, dst_video_img_dir)
                print(f'Скопировано изображений: {video_id} ({len(os.listdir(src_video_img_dir))} файлов)')

            # Создаем папку для меток
            label_video_dir = os.path.join(LABEL_DIR, video_id)
            os.makedirs(label_video_dir, exist_ok=True)

            frame_count = 0
            with open(ann_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    try:
                        frame_id, _, x, y, w, h, cls, *_ = map(int, parts[:8])
                        if cls >= 10:  # Пропускаем классы >= 10
                            continue

                        img_filename = f"{frame_id:07d}.jpg"
                        img_path = os.path.join(dst_video_img_dir, img_filename)
                        label_path = os.path.join(label_video_dir, f"{frame_id:07d}.txt")

                        if not os.path.exists(img_path):
                            continue

                        try:
                            with Image.open(img_path) as img:
                                img_w, img_h = img.size
                        except:
                            continue

                        # Конвертируем в формат YOLO (нормализованные координаты)
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h

                        # Проверяем, что координаты в допустимых пределах
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1:
                            with open(label_path, 'a') as f_out:
                                f_out.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                            frame_count += 1

                    except (ValueError, IndexError) as e:
                        continue

            if frame_count > 0:
                print(f'Сконвертировано: {ann_file} ({frame_count} кадров)')
                converted_count += 1
            else:
                print(f'Пропущено: нет валидных данных в {ann_file}')
                skipped_count += 1

        print(f"\nИтого для {data_type}:")
        print(f"Сконвертировано файлов: {converted_count}")
        print(f"Пропущено файлов: {skipped_count}")

    print(f"\n{'=' * 50}")
    print(f"Структура данных создана в: {DATASET_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    convert_annotations_server()