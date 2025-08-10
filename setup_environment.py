#!/usr/bin/env python3
"""
Скрипт для настройки виртуального окружения на сервере
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Выполняет команду с выводом прогресса"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при {description}: {e}")
        print(f"Вывод ошибки: {e.stderr}")
        return False

def setup_virtual_environment():
    """Настраивает виртуальное окружение для проекта"""
    print("🚀 === НАСТРОЙКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ ===")
    print("=" * 50)
    
    # Базовая директория
    BASE_DIR = os.path.expanduser('~/Bespilot_lopatinBeglov')
    VENV_DIR = os.path.join(BASE_DIR, 'venv')
    
    print(f"📁 Базовая директория: {BASE_DIR}")
    print(f"🐍 Виртуальное окружение: {VENV_DIR}")
    
    # Создаем базовую директорию
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Проверяем наличие Python
    print(f"\n🔍 Проверка Python...")
    python_version = subprocess.run([sys.executable, '--version'], 
                                  capture_output=True, text=True).stdout.strip()
    print(f"✅ {python_version}")
    
    # Создаем виртуальное окружение
    if not run_command(f"python -m venv {VENV_DIR}", "Создание виртуального окружения"):
        return False
    
    # Активируем виртуальное окружение и обновляем pip
    activate_script = os.path.join(VENV_DIR, 'bin', 'activate')
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(VENV_DIR, 'Scripts', 'activate.bat')
    
    # Обновляем pip
    pip_cmd = f"source {VENV_DIR}/bin/activate && pip install --upgrade pip"
    if os.name == 'nt':
        pip_cmd = f"{VENV_DIR}\\Scripts\\activate && pip install --upgrade pip"
    
    if not run_command(pip_cmd, "Обновление pip"):
        return False
    
    # Устанавливаем зависимости
    requirements_path = os.path.join(BASE_DIR, 'requirements.txt')
    if os.path.exists(requirements_path):
        install_cmd = f"source {VENV_DIR}/bin/activate && pip install -r {requirements_path}"
        if os.name == 'nt':
            install_cmd = f"{VENV_DIR}\\Scripts\\activate && pip install -r {requirements_path}"
        
        if not run_command(install_cmd, "Установка зависимостей"):
            return False
    else:
        print("⚠️  Файл requirements.txt не найден, устанавливаем основные пакеты...")
        basic_packages = "ultralytics torch torchvision Pillow PyYAML tqdm opencv-python matplotlib seaborn pandas numpy"
        install_cmd = f"source {VENV_DIR}/bin/activate && pip install {basic_packages}"
        if os.name == 'nt':
            install_cmd = f"{VENV_DIR}\\Scripts\\activate && pip install {basic_packages}"
        
        if not run_command(install_cmd, "Установка основных пакетов"):
            return False
    
    # Создаем скрипт активации
    print(f"\n📄 Создание скрипта активации...")
    activate_script_content = f"""#!/bin/bash
# Скрипт активации виртуального окружения для проекта Bespilot_lopatinBeglov
echo "🚀 Активация виртуального окружения..."
source {VENV_DIR}/bin/activate
echo "✅ Виртуальное окружение активировано!"
echo "📁 Текущая директория: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Версия ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'не установлен')"
"""
    
    if os.name == 'nt':
        activate_script_content = f"""@echo off
REM Скрипт активации виртуального окружения для проекта Bespilot_lopatinBeglov
echo 🚀 Активация виртуального окружения...
call {VENV_DIR}\\Scripts\\activate.bat
echo ✅ Виртуальное окружение активировано!
echo 📁 Текущая директория: %cd%
echo 🐍 Python: %where python%
"""
    
    activate_file = os.path.join(BASE_DIR, 'activate_env.sh' if os.name != 'nt' else 'activate_env.bat')
    with open(activate_file, 'w') as f:
        f.write(activate_script_content)
    
    # Делаем скрипт исполняемым (для Linux/Mac)
    if os.name != 'nt':
        os.chmod(activate_file, 0o755)
    
    print(f"✅ Создан скрипт активации: {activate_file}")
    
    # Создаем инструкцию
    print(f"\n📋 Создание инструкции по использованию...")
    instruction_content = f"""# Инструкция по использованию виртуального окружения

## Активация окружения

### Linux/Mac:
```bash
cd ~/Bespilot_lopatinBeglov
source activate_env.sh
```

### Windows:
```cmd
cd ~/Bespilot_lopatinBeglov
activate_env.bat
```

## Ручная активация

### Linux/Mac:
```bash
source ~/Bespilot_lopatinBeglov/venv/bin/activate
```

### Windows:
```cmd
~\\Bespilot_lopatinBeglov\\venv\\Scripts\\activate
```

## Проверка установки

После активации окружения выполните:
```bash
python -c "import ultralytics; print('Ultralytics установлен:', ultralytics.__version__)"
python -c "import torch; print('PyTorch установлен:', torch.__version__)"
python -c "import tqdm; print('tqdm установлен')"
```

## Запуск проекта

1. Активируйте окружение
2. Перейдите в папку со скриптами:
   ```bash
   cd ~/Bespilot_lopatinBeglov/scripts
   ```
3. Запустите скрипты по порядку:
   ```bash
   python check_dataset.py
   python convert_annotations_server.py
   python prepare_server_dataset.py
   python train_server.py
   ```

## Деактивация

```bash
deactivate
```
"""
    
    instruction_file = os.path.join(BASE_DIR, 'VENV_INSTRUCTIONS.md')
    with open(instruction_file, 'w') as f:
        f.write(instruction_content)
    
    print(f"✅ Создана инструкция: {instruction_file}")
    
    print(f"\n🎉 Настройка виртуального окружения завершена!")
    print(f"📁 Виртуальное окружение: {VENV_DIR}")
    print(f"📄 Скрипт активации: {activate_file}")
    print(f"📋 Инструкция: {instruction_file}")
    
    print(f"\n🚀 Следующие шаги:")
    print(f"1. Активируйте окружение: source {activate_file}")
    print(f"2. Скопируйте данные в {BASE_DIR}")
    print(f"3. Запустите скрипты подготовки данных")

if __name__ == "__main__":
    setup_virtual_environment()
