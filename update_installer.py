import requests
import os
import subprocess
import sys
import psutil
from dotenv import load_dotenv
from notifypy import Notify

load_dotenv()


def show_notification(title, message, duration=10):
    notification = Notify()
    notification.application_name = "KeywordCraze"
    notification.title = title
    notification.message = message
    notification.icon = "assets/favicon.ico"
    notification.duration = duration
    notification.send()

def get_version():
    with open("version.txt", "r") as file:
        version = file.read().strip().split("\n")[0]
    print(f"Current version: {version}")
    return version
def update_version():
    download_link = None
    with open("version.txt", "r") as file:
        print(f"File content: {file.read()}, latest version: {LATEST_VERSION}")
        download_link = file.read().strip().split("\n")[1]
    with open("version.txt", "w") as file:
        file.write(f"{LATEST_VERSION}\n{download_link}")
    print(f"Version updated successfully to {LATEST_VERSION}")

# Получаем путь к AppData текущего пользователя
REMOTE_APP_URL = os.getenv("REMOTE_APP_URL")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_VERSION = get_version()
VERSION_URL = f"{REMOTE_APP_URL}version.txt"
LATEST_VERSION = None

print(f"App dir: {APP_DIR}")

def close_current_application():
    """Функция закрывает текущий процесс приложения."""
    try:
        print("Closing current application...")
        for proc in psutil.process_iter():
            try:
                # Проверяем имя процесса
                if "KeywordCraze.exe" in proc.name():
                    proc.terminate()  # Завершаем процесс
                    proc.wait(timeout=3)
                    print("Application closed successfully.")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"Error closing the application: {e}")

def check_for_updates():
    global LATEST_VERSION
    try:
        response = requests.get(VERSION_URL)
        if response.status_code == 200:
            print(f"response: {response.text}")
            latest_version, download_link = response.text.strip().split("\n")
            if latest_version > CURRENT_VERSION:
                print(f"New version available: {latest_version}")
                LATEST_VERSION = latest_version
                return True, download_link
            else:
                show_notification("Keyword Craze", "You have the latest version.")
                return False, None
    except Exception as e:
        show_notification("Error", f"Error checking for updates: {e}")
    return False, None

def download_update(download_url, save_path=f"{APP_DIR}"):
    try:
        print("Downloading update...")
        print(f"Save path: {save_path}")
        response = requests.get(download_url, stream=True)
        print(f"Downloading response: {response}")
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Update downloaded to {save_path}")
        return save_path
    except Exception as e:
        show_notification("Error", f"Error downloading update: {e}")
    return None

def install_update(installer_path):
    try:
        print("Installing update...")

        # Закрываем приложение
        close_current_application()

        # Запускаем вспомогательный скрипт для замены файла
        helper_path = os.path.join(APP_DIR, "update_helper.exe")
        subprocess.Popen([helper_path, installer_path, "KeywordCraze.exe"], shell=False)

        print("Update installation started. Exiting updater...")
        sys.exit(0)
    except Exception as e:
        show_notification("Error", f"Error installing update: {e}")

# Основная логика
if __name__ == "__main__":
    print("Checking for updates...")
    update_available, link = check_for_updates()
    if update_available:
        installer = download_update(link)
        if installer:
            install_update(installer)
    else:
        print("No updates available.")
