import zipfile
import shutil
from tkinter import filedialog, Tk, Button, Label, Frame, Scrollbar, Canvas
from PIL import Image, ImageEnhance, ImageTk
import os
from openai import OpenAI
import base64
import threading
import subprocess
# import onnxruntime as ort
# import numpy 
from notifypy import Notify
from dotenv import load_dotenv
import datetime
import queue
import json
import openpyxl 
import sys 
import atexit


load_dotenv()

# Получаем путь к AppData текущего пользователя
appdata_dir = os.getenv("APPDATA")
base_dir = os.path.join(appdata_dir, "KeywordCraze")  # Главная директория программы в AppData

# Путь к папкам
# resources_folder = "resources"
resources_folder = os.path.join(base_dir, "resources")
zip_folder = os.path.join(base_dir, "archives")

# Создаем имена файлов с датой
today_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
zip_file_name = f"images_keywords_{today_date}.zip"
excel_file_name = f"images_keywords_{today_date}.xlsx"
zip_file_path = os.path.join(zip_folder, zip_file_name)
excel_file_path = os.path.join(zip_folder, excel_file_name)

# Списки и переменные
uploaded_images = []
processed_images = []
processed_images_with_metadata = []
task_queue = queue.Queue()
pending_tasks = 0
task_lock = threading.Lock()
stop_event = threading.Event()

# Создаем Excel-файл
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = f"Images Metadata {today_date}"

# Записываем заголовки в Excel
sheet['A1'] = "Image Name"
sheet['B1'] = "Title"
sheet['C1'] = "Keywords"

# Функция для создания папок
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Создаем необходимые папки
create_folder(base_dir)
create_folder(resources_folder)
create_folder(zip_folder)

# Инициализация клиента OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Удаляем папку resources, если она существует
if os.path.exists(resources_folder):
    shutil.rmtree(resources_folder)  # Удаляет папку и всё её содержимое

# Создаём папку resources заново
os.makedirs(resources_folder)

# Функция для остановки всех потоков
def stop_threads():
    print("Завершаем работу программы...")
    stop_event.set()  # Сигнализируем всем потокам завершить работу

# Добавление обработчика завершения программы
atexit.register(stop_threads)

def show_notification(title, message, duration=20):
    notification = Notify()
    notification.application_name = "KeywordCraze"
    notification.title = title
    notification.message = message
    notification.icon = "assets/favicon.ico"
    notification.duration = duration
    notification.send()


def add_image(img_path):
    if img_path not in uploaded_images:  # Проверяем, чтобы не добавлять дубликаты
        uploaded_images.append(img_path)
    display_images()


def clear_images():
    uploaded_images.clear()
    display_images()
    progress_label.config(text=f"Status: {len(uploaded_images)} images uploaded")


# Функция для загрузки изображений
def load_images():
    filepaths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    for filepath in filepaths:
        # Добавляем изображение в список выбранных
        add_image(filepath)
    display_images()
    progress_label.config(text=f"Status: {len(uploaded_images)} images uploaded")
    


# Функция для отображения изображений
def display_images(images=None):
    # Очищаем предыдущие миниатюры
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    if images or uploaded_images:
        clear_button.pack(side="left", padx=5)
        process_button.pack(side="left", padx=5)
    else:
        clear_button.pack_forget()
        process_button.pack_forget()

    row = 0  # Стартовая строка для grid
    col = 0  # Стартовый столбец для grid

    canvas_frame.grid_columnconfigure(0, weight=1, uniform="equal")
    canvas_frame.grid_rowconfigure(0, weight=1, uniform="equal")
    if images is None:
        # Отображаем каждое изображение с кнопкой для удаления
        for img_path in uploaded_images:
            img = Image.open(img_path)
            img.thumbnail((150, 150))  # Устанавливаем размер миниатюры
            img_tk = ImageTk.PhotoImage(img)

            
            # Создаем Frame для изображения и кнопки
            img_frame = Frame(canvas_frame)
            img_frame.grid(row=row, column=col, columnspan=1, rowspan=1, padx=5, pady=5, sticky="nsew")

            # Создаем метку для миниатюры
            label = Label(img_frame, image=img_tk)
            label.image = img_tk  # Сохраняем ссылку на изображение, чтобы оно не исчезло
            label.pack(side="top", pady=5)

            # Создаем кнопку для удаления изображения
            remove_button = Button(img_frame, text="Remove", command=lambda img=img_path: remove_image(img), bg="#ee6055", fg="white")
            remove_button.pack(side="bottom", pady=5)
            # Перемещаемся на следующий столбец
            col += 1

            # Если столбцы достигли максимума (например, 4), то переходим на новую строку
            if col > 4:
                col = 0
                row += 1
    else:
        for image_data in images:
            img_path = image_data["image_path"]
            title = image_data["title"]
            keywords = image_data["keywords"]

            img = Image.open(img_path)
            img.thumbnail((200, 200))  # Устанавливаем размер миниатюры
            img_tk = ImageTk.PhotoImage(img)

            # Создаем Frame для изображения
            img_frame = Frame(canvas_frame)
            img_frame.grid(row=row, column=col, columnspan=1, rowspan=1, padx=5, pady=5, sticky="nsew")

            # Создаем метку для миниатюры
            label = Label(img_frame, image=img_tk)
            label.image = img_tk  # Сохраняем ссылку на изображение
            label.pack(side="top", pady=5)

            # Размещаем информацию справа от изображения
            info_frame = Frame(canvas_frame)
            info_frame.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")

            # Заголовок
            title_label = Label(info_frame, text=title, anchor="w", font=("Arial", 15, "bold"), wraplength=600, justify="left")
            title_label.pack(side="top", anchor="w")

            # Ключевые слова
            keywords_label = Label(info_frame, text="\nKeywords: \n" + keywords, anchor="w" , font=("Arial", 12), wraplength=600, justify="left")
            keywords_label.pack(side="top", anchor="w")

            # Перемещаемся на следующий ряд
            row += 1
        
    # for i in range(col + 1):
    #     canvas_frame.grid_columnconfigure(i, weight=1, uniform="equal")  # Применяем для всех столбцов

# Функция для удаления изображения из списка
def remove_image(img_path):
    if img_path in uploaded_images:
        uploaded_images.remove(img_path)  # Удаляем изображение из списка
    display_images()  # Обновляем отображение
    progress_label.config(text=f"Status: {len(uploaded_images)} images uploaded")

def update_image_list():
    # Очищаем старые миниатюры
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Обновляем отображение всех выбранных изображений
    for image_path in uploaded_images:
        img = Image.open(image_path)
        img.thumbnail((100, 100))  # Уменьшаем размер изображения для миниатюры
        img_tk = ImageTk.PhotoImage(img)

        label = Label(canvas_frame, image=img_tk)
        label.image = img_tk  # Необходимо сохранить ссылку на изображение
        label.pack(side="left", padx=5)
    progress_label.config(text=f"Status: {len(uploaded_images)} images uploaded")


def upload_images():
    print("Select images to upload")
    progress_label.config(text=f"Status: uploading images...")
    copied_filepaths = []

    # Функция для обработки изображений
    def process_images():
        for filepath in uploaded_images:
            filename, ext = os.path.splitext(os.path.basename(filepath))
            destination = os.path.join(resources_folder, f"{filename}.jpg")  # Устанавливаем расширение .jpg для сохранения
            try:
                img = Image.open(filepath)
                if img.mode != "RGB":
                    print(f"Converting image {filename} to RGB mode...")
                    img = img.convert("RGB")
                img.save(destination, format="JPEG", quality=85, optimize=True)  # Используем качество 85 для меньшего размера файла
                copied_filepaths.append(destination)
                print(f"Image {filename} converted to JPEG and saved to {destination}") 
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Запуск обработки изображений в отдельном потоке
    thread = threading.Thread(target=process_images, daemon=True)
    thread.start()
    thread.join()
    print("Image processing completed.")
    progress_label.config(text=f"Status: {len(copied_filepaths)} images uploaded")

    return copied_filepaths

def local_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    # Convert the image to Base64
    encoded_image = base64.b64encode(image_data)
    base64_string = encoded_image.decode("utf-8")
    return base64_string
# # return f"data:image/jpeg;base64,{encoded_image}"  # Указываем MIME-тип (например, для jpeg)

def analyze_images_with_gpt(image_url):
    print("Analyzing images with GPT...")

    base64_string = local_image_to_base64(image_url)


    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1,
        messages=[{
                "role": "system",
                "content": "Generate 45 adobe stock image keywords and a short description for the given image. Return the response in format: {\"keywords\": \"comma-separated keywords\", \"description\": \"short description 10-15 words\"}. Only return this JSON with no extra text."
            },
            {
                "role": "user",
                # "content": [{"type": "image_url", "image_url": {"url": f"{image_url}"}}]
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"}}]
            }
        ]
    )
    print(f"Response: {response}")
    response_content = response.choices[0].message.content
    
    response_json = json.loads(response_content)  # Parse as JSON
    title = response_json.get('description', '')
    keywords = [keyword.strip().lower() for keyword in response_json.get('keywords', '').split(', ')]
    # title = "A cheerful girl holding a gift, adorned in winter clothing, celebrating the festive season."
    # keywords = "christmas, girl, gift, illustration, winter, holiday, smiling, winterwear, orange, scarf, cartoon, tree, festive, present, happy, childhood, cute, snow, celebration, pine, joyful, decoration, wrapped gift, gloves, character, seasonal, hand-drawn, cozy, cheerful, adorable, family, happiness, fun, childhood memories, nature, evergreen, branches, festive season, surprise, anticipation, warmth".split(', ')
    print(f"Title: {title}\n")
    print(f"Keywords: {keywords}\n\n")
    return title, keywords
    
# Функция для добавления ключевых слов в метаданные изображения
def update_image_metadata(image_path, title, keywords):
    print(f"Updating metadata for {image_path}...")
     # Путь к exiftool в режиме сборки через PyInstaller
    if getattr(sys, 'frozen', False):  # Проверка, запущено ли приложение как EXE
        base_path = sys._MEIPASS  # PyInstaller распаковывает файлы сюда
    else:
        base_path = os.path.dirname(__file__)  # В режиме обычного Python-скрипта
    print(f"Base path: {base_path}")
    # Path to exiftool.exe
    exiftool_path = os.path.join(base_path, 'assets', 'exiftool.exe')
    # image_full_path = os.path.join(os.path.dirname(__file__), image_path)
    image_full_path = os.path.abspath(image_path)

    # Validate paths
    if not os.path.exists(exiftool_path):
        print(f"ExifTool not found at {exiftool_path}")
        raise show_notification("Error", f"ExifTool not found at {exiftool_path}")
        # raise FileNotFoundError(f"ExifTool not found at {exiftool_path}")
    if not os.path.exists(image_full_path):
        print(f"Image file not found at {image_full_path}")
        raise show_notification("Error", f"Image file not found at {image_full_path}")
        # raise FileNotFoundError(f"Image file not found at {image_full_path}")

    # Prepare command
    cmd = [
        exiftool_path,
        '-overwrite_original',      
        '-largetags',
    ]

    # Add title-related fields
    title_fields = ['-Title', '-Description', '-Caption-Abstract', '-Headline']
    cmd.extend([f'{field}={title}' for field in title_fields])

    # Add keywords
    if isinstance(keywords, list):
        keyword_flags = [f'-Keywords={kw}' for kw in keywords]
        cmd.extend(keyword_flags)
    else:
        cmd.append(f'-Keywords={keywords}')

    # Add the image path
    cmd.append(image_full_path)

    # Execute the command
    try:
        subprocess.run(cmd, check=True, shell = "False")
        print(f"Metadata updated for {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating metadata: {e}")

    # progress_label.config(text=f"Status: Adjusting image size for {image_path}")




# def upscale_image_realesrganx4(image_path, output_path):
#     # Load the ONNX model
#     model_path = 'assets/models/RealESR_Gx4_fp16.onnx'
#     session = ort.InferenceSession(model_path)

#     # Load and preprocess the image
#     image = Image.open(image_path).convert('RGB')  # Ensure RGB format
#     image = numpy.array(image).astype(numpy.float32)

#     # Normalize and prepare input
#     image = image / 255.0  # Normalize to [0, 1]
#     h, w, _ = image.shape
#     new_h = (h // 4) * 4  # Ensure height is divisible by 4
#     new_w = (w // 4) * 4  # Ensure width is divisible by 4
#     image = image[:new_h, :new_w, :]  # Crop image if necessary
#     image = numpy.transpose(image, (2, 0, 1))  # HWC to CHW
#     image = numpy.expand_dims(image, axis=0)  # Add batch dimension (NCHW)

#     # Get model input name
#     input_name = session.get_inputs()[0].name

#     # Run inference
#     output = session.run(None, {input_name: image})

#     # Post-process the output
#     output_image = output[0].squeeze()  # Remove batch dimension
#     output_image = numpy.transpose(output_image, (1, 2, 0))  # CHW to HWC
#     output_image = (output_image * 255.0).clip(0, 255).astype(numpy.uint8)  # De-normalize

#     # Save the output image
#     output_image = Image.fromarray(output_image)
#     output_image.save(output_path)

#     print(f"Upscaled image saved to {output_path}")

def adjust_image_size(image_path, target_min_mb=5, target_max_mb=12):
    progress_label.config(text=f"Status: Adjusting image size...")
    print(f"Adjusting image size for {image_path}...")

    # Открываем изображение
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Начальные параметры
    upscale_factor = 1.5  # Начальный коэффициент увеличения
    quality = 100  # Стартовое качество
    min_quality = 10  # Минимальное качество
    temp_path = image_path.replace(".jpg", "_temp.jpg")
    
    # Удаляем временный файл, если он существует
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Основной цикл для регулировки размера файла
    while True:
        # Увеличиваем разрешение
        new_width = int(original_width * upscale_factor)
        new_height = int(original_height * upscale_factor)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Улучшаем резкость
        enhancer = ImageEnhance.Sharpness(resized_img)
        enhanced_img = enhancer.enhance(2)

        # Сохраняем изображение с текущим качеством в памяти
        from io import BytesIO
        byte_io = BytesIO()
        enhanced_img.save(byte_io, format="JPEG", quality=quality, optimize=True, progressive=True)

        # Получаем размер изображения в памяти
        file_size = byte_io.tell() / (1024 * 1024)  # В MB

        print(f"File size: {file_size:.2f} MB for {temp_path}")

        # Если размер файла в целевом диапазоне, сохраняем на диск
        if target_min_mb <= file_size <= target_max_mb:
            with open(temp_path, 'wb') as f:
                byte_io.seek(0)
                f.write(byte_io.read())
            break
        # Если файл меньше минимального размера, увеличиваем разрешение
        elif file_size < target_min_mb:
            upscale_factor += 1
        # Если файл больше максимального размера, уменьшаем качество
        elif file_size > target_max_mb:
            quality = max(min_quality, quality - 2)

    # Заменяем исходное изображение на новое
    if os.path.exists(image_path):
        os.remove(image_path)
    os.rename(temp_path, image_path)
    print(f"Image size adjusted to {file_size:.2f} MB for {image_path}")
    return image_path

def resize_to_fit(image_path, output_path, min_width=1350, min_height=1000):
    with Image.open(image_path) as img:
        if img.width > min_width and img.height > min_height:
            print(f"Image {image_path} is already larger than {min_width}x{min_height}, skipping resize.")
        else: 
            # Вычисляем новый размер
            scale = max(min_width / img.width, min_height / img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(output_path)
            print(f"Изображение масштабировано до {new_width}x{new_height} и сохранено в {output_path}")

# Функция для создания ZIP-архива из изображений
def create_zip_archive(images, zip_filename):
    # Создаём папку resources заново
    print("Creating ZIP archive...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img in images:
            zipf.write(img, os.path.basename(img))
    print("ZIP archive created successfully!")
    # with zipfile.ZipFile(zip_filename, 'r', zipfile.ZIP_DEFLATED) as zipf:
    #      for file_info in zipf.infolist():
    #           print(f"{file_info.filename}: {file_info.file_size / (1024 * 1024):.2f} MB in ZIP")


def run_task_in_background(task_func, *args):
    """Function to run tasks in background using threading."""
    def task_wrapper():
        try:
            task_func(*args)  # Run the actual task
        except Exception as e:
            progress_label.config(text=f"Error: {e}")
        finally:
            root.after(0, update_gui_after_task)  # Update GUI after task completion
    
    # Start the task in a separate background thread
    threading.Thread(target=task_wrapper, daemon=True).start()

def update_gui_after_task():
    """Function to update the GUI after background task completion."""
    if pending_tasks == 0:
        progress_label.config(text=f"Status: All tasks completed!")
    else:
        progress_label.config(text=f"Status: Pending tasks: {pending_tasks}")
# def process_single_image(filepath):

#     progress_label.config(text=f"Status: Resizing images to fit...")
#     resize_to_fit(filepath, filepath)
#     # cloudinary_images.append(upload_image_to_cloudinary(filepath))
#     # keywords, description = generate_metadata(filepath)
#     # title, keywords = run_task_in_background(analyze_images_with_gpt, filepath)
#     progress_label.config(text=f"Status: Generating metadata for {filepath}...")
#     title, keywords = analyze_images_with_gpt(filepath) 
        
#     # upscale_image_realesrganx4(filepath, filepath)
#     # model_path = "assets/models/RealESR_Gx4_fp16.onnx"  # Adjust model path
#     # upscale_image(filepath, model_path, "temp", num_threads=2)

#     # add_metadata_to_image(filepath, keywords, description)
#     progress_label.config(text=f"Status: Updating metadata for {filepath}...")
#     # run_task_in_background(update_image_metadata, filepath, title, keywords)
#     update_image_metadata(filepath, title, keywords)
#     # Добавляем обработанное изображение в список
#     processed_images_with_metadata.append({"image_path": filepath, 'title': title, 'keywords': ", ".join(keywords)})
#     processed_images.append(filepath)

#     print(f"Processed image: {filepath}")

#     progress_label.config(text=f"Finished processing {filepath}")

def process_single_image(filepath):
    """Processing a single image in background, updating progress after each task."""
    progress_label.config(text=f"Status: Resizing image {filepath}...")
    global pending_tasks
    global processed_images 
    global processed_images_with_metadata

    # Here, we use threading for each step to make sure they don't block the UI
    resize_thread = threading.Thread(target=lambda: resize_to_fit(filepath, filepath), daemon=True)
    resize_thread.start()
    resize_thread.join()  # Wait for resize to finish
    
    
    progress_label.config(text=f"Status: Generating metadata...")
    title, keywords = analyze_images_with_gpt(filepath)  # Simulate external API processing
    adjust_image_thread = threading.Thread(target=lambda: adjust_image_size(filepath), daemon=True)
    adjust_image_thread.start()
    adjust_image_thread.join()  # Wait for resize to finish
    update_image_metadata(filepath, title, keywords)  # Update image metadata

    # Add to processed images list after processing
    processed_images.append(filepath)
    processed_images_with_metadata.append({"image_path": filepath, 'title': title, 'keywords': ", ".join(keywords)})
    print(f"Pending tasks: {pending_tasks}")

    print(f"Processed image: {filepath}")
    # After image is processed, update the status
    decrement_pending_tasks()
    root.after(0, lambda: print(f"Finished processing {filepath}"))

def decrement_pending_tasks():
    """Decrement the pending tasks counter and trigger ZIP creation if done."""
    global pending_tasks
    print(f"Decrementing pending tasks to {pending_tasks}")
    with task_lock:
        pending_tasks -= 1  # Decrease task count
        if pending_tasks == 0:
            # All tasks are done, create ZIP archive
            root.after(0, create_zip_after_processing)



# Основная функция для обработки изображений и создания архива
def process_and_zip_images():
    global pending_tasks, processed_images, processed_images_with_metadata
    zip_folder_button.pack_forget()
    download_zip_button.pack_forget()
    download_excel_button.pack_forget()
    print("Processing and zipping images...")
    progress_label.config(text="Status: Processing images...")
    filepaths = upload_images()
    pending_tasks = len(filepaths)
    processed_images = []
    processed_images_with_metadata = []
    progress_label.config(text="Status: Processing images...")
    
    # Process images in background
    for filepath in filepaths:
        run_task_in_background(process_single_image, filepath)


def create_zip_after_processing():
    """Create zip file once all images are processed."""
    print("All images processed. Creating ZIP archive...")
    progress_label.config(text="Status: Creating ZIP archive...")
    create_zip_archive(processed_images, zip_file_path)  # Create the ZIP file
    progress_label.config(text="Status: ZIP archive created")
    create_excel_file()
    # Show the download button after archive is created
    zip_folder_button.pack(side="left", padx=5, pady=5)
    download_zip_button.pack(side="left", padx=5, pady=5)
    download_excel_button.pack(side="left", padx=5, pady=5)
    display_images(processed_images_with_metadata)

    show_notification("Keyword Craze", f"ZIP archive created successfully! Download it now.")
def create_excel_file():
    global excel_file_path, processed_images_with_metadata
    data = []
    for image_data in processed_images_with_metadata:
        data.append([image_data['image_path'].split("\\")[-1], image_data['title'], image_data['keywords']])
    for row in data:
        sheet.append(row)
    workbook.save(excel_file_path)
    print("Excel file created successfully!")

def download_zip():
    if os.path.exists(zip_file_path):
        # Открываем диалог для сохранения архива в выбранное место
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP files", "*.zip")], initialfile=zip_file_name)
        if save_path:
            try:
                # Копируем архив в выбранное место
                shutil.copy(zip_file_path, save_path)
                progress_label.config(text=f"Status: ZIP archive downloaded to {save_path}")
                show_notification("Keyword Craze", f"ZIP archive downloaded to {save_path}")
            except Exception as e:
                show_notification("Error", f"Error downloading ZIP archive: {e}")
    else:
        show_notification("No ZIP", "No ZIP archive has been created yet!")


def download_excel():
    global excel_file_path, excel_file_name
    if os.path.exists(excel_file_path):
        # Открываем диалог для сохранения архива в выбранное место
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("EXCEL files", "*.xlsx")], initialfile=excel_file_name)
        if save_path:
            try:
                # Копируем архив в выбранное место
                shutil.copy(excel_file_path, save_path)
                progress_label.config(text=f"Status: Excel file downloaded to {save_path}")
                show_notification("Keyword Craze", f"Excel file downloaded to {save_path}")
            except Exception as e:
                show_notification("Error", f"Error downloading Excel file: {e}")
                # messagebox.showerror("Error", f"Error downloading Excel file: {e}")
    else:
        show_notification("No Excel", "No Excel file has been created yet!")

# Функция для открытия папки с архивом
def open_zip_folder():
    # Получаем путь к корневой директории проекта
    project_root = os.path.dirname(os.path.abspath(__file__))  # Получаем путь к текущей директории (корень проекта)

    # Папка с архивом в корне проекта
    folder_path = os.path.join(project_root, zip_folder)  # Замените "zip_folder" на название вашей папки

    # В Windows
    try:
        os.startfile(folder_path)  # Открывает папку в проводнике
    except Exception as e:
        print(f"Error opening folder: {e}")

# bg_color = "#AAD7D9"
bg_color = "#f8f9fa"

# Основное окно Tkinter
root = Tk()
root.iconbitmap('assets/favicon.ico')
root.title("Adobe Stock Keywords Generator")
root.geometry("900x700")
root.configure(padx=20, pady=20)

# Метка для отображения прогресса
progress_label = Label(root, text="Status: Waiting for input",  font=("Arial", 12, "bold"), )
progress_label.pack(pady=20)

main_button_frame = Frame(root, )
main_button_frame.pack(pady=10, side="top")  
button_frame = Frame(main_button_frame, )
button_frame.pack(pady=10, side="top")  
second_button_frame = Frame(main_button_frame, )
second_button_frame.pack(pady=20, side="top")  
# Кнопка для загрузки изображений
upload_button = Button(button_frame, text="Upload Images", command=load_images, font=("Arial", 12, "bold"), bg="#212529", fg="white")
upload_button.pack(side="left", padx=5)  # Кнопка загрузки изображений
# Кнопка для загрузки изображений
clear_button = Button(button_frame, text="Clear Images", command=clear_images, font=("Arial", 12, "bold"), bg="#ee6055", fg="white")
# clear_button.pack(side="left", padx=5,) 
clear_button.pack_forget() 

process_button = Button(button_frame, text="Start Processing", command=process_and_zip_images, font=("Arial", 12, "bold"), bg="#212529", fg="white")
process_button.pack_forget()  # Кнопка для начала обработки



# Кнопка для открытия папки
zip_folder_button = Button(second_button_frame, text="Open archive folder", command=open_zip_folder, font=("Arial", 12, "bold"), bg="#212529", fg="white")
zip_folder_button.pack_forget()
# Кнопка для скачивания архива
download_zip_button = Button(second_button_frame, text="Download ZIP", command=download_zip, font=("Arial", 16, "bold"), bg="#06d6a0")
download_zip_button.pack_forget()
# Кнопка для скачивания excel
download_excel_button = Button(second_button_frame, text="Download Excel", command=download_excel, font=("Arial", 12, "bold"), bg="#212529", fg="white")
download_excel_button.pack_forget()

# Создаем основной фрейм
scroll_frame = Frame(root)
scroll_frame.pack(expand=True, fill="both")

# Создаем Canvas
canvas = Canvas(scroll_frame, highlightthickness=0)
canvas.pack(side="left", expand=True, fill="both")

# Создаем вертикальный Scrollbar и привязываем его к Canvas
scrollbar = Scrollbar(scroll_frame, orient="vertical", command=canvas.yview,)
scrollbar.pack(side="right", fill="y")

# Привязка прокрутки к Canvas
canvas.configure(yscrollcommand=scrollbar.set)

# Создаем Frame внутри Canvas, куда будут добавляться элементы
canvas_frame = Frame(canvas)
canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

# Обновляем размер Canvas при изменении содержимого
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

canvas_frame.bind("<Configure>", on_frame_configure)


# Функция для прокрутки колесиком мыши
def on_mouse_wheel(event):
    # Проверяем направление прокрутки
    if event.delta > 0:  # Прокрутка вверх
        canvas.yview_scroll(-1, "units")
    else:  # Прокрутка вниз
        canvas.yview_scroll(1, "units")

# Привязываем событие колесика мыши
canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Запуск интерфейса
root.mainloop()
