import os
import sys
import time
import shutil
import subprocess

def replace_and_restart(source_file, target_file, executable_to_run):
    try:
        # Пауза для завершения KeywordCraze.exe
        print("Waiting for the application to close...")
        time.sleep(3)

        # Заменяем файл
        print(f"Replacing {target_file} with {source_file}...")
        shutil.move(source_file, target_file)

        # Запускаем обновлённое приложение
        print("Restarting the updated application...")
        subprocess.Popen([executable_to_run], shell=False)
    except Exception as e:
        print(f"Error during update: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    # Путь к файлам
    source_file = sys.argv[1]  # Временный файл обновления
    target_file = sys.argv[2]  # Путь к KeywordCraze.exe
    executable_to_run = target_file

    replace_and_restart(source_file, target_file, executable_to_run)
