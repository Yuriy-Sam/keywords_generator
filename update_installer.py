import requests
import os
import subprocess
import sys
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
CURRENT_VERSION = get_version()
VERSION_URL = f"{REMOTE_APP_URL}version.txt"
LATEST_VERSION = None




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
                subprocess.Popen("KeywordCraze.exe", shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
                return False, None
    except Exception as e:
        show_notification("Error", f"Error checking for updates: {e}")
    return False, None

def download_update(download_url, save_path="KeywordCraze.exe"):
    try:
        print("Downloading update...")
        response = requests.get(download_url, stream=True)
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
        subprocess.Popen(installer_path, shell=True)
        print("Update installed successfully.")
        update_version()
        show_notification("Keyword Craze", "Update installed successfully.")
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
