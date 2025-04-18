import pyaudio  # Для работы с аудиовходом (микрофон) и аудиовыходом (воспроизведение)
import wave     # Для чтения и записи аудиофайлов в формате WAV
import time     # Для работы со временем
import os       # Для взаимодействия с операционной системой (пути к файлам, переменные окружения, запуск файлов)
import webbrowser # Для открытия веб-браузера (для поиска в интернете)
import struct   # Для упаковки/распаковки бинарных данных (используется для аудиоданных Porcupine)
import pvporcupine # Основная библиотека для детекции ключевого слова (wake word) от Picovoice
import re       # Для работы с регулярными выражениями (поиск и обработка текста)
import requests # Для выполнения HTTP-запросов (для общения с API Hugging Face)
import json     # Для работы с данными в формате JSON (чтение/запись лог-файла)
import ctypes   # Для взаимодействия с C-совместимыми библиотеками (используется для вызова Windows API)
from num2words import num2words # Для преобразования чисел в слова (для озвучки чисел TTS)
from ctypes import wintypes, cast, POINTER # Специфичные типы и функции из ctypes для работы с Windows API
from comtypes import CLSCTX_ALL # Константа для COM-взаимодействия (используется pycaw для управления громкостью)
import whisper # Библиотека для распознавания речи (Speech-to-Text)
import torch   # Основная библиотека машинного обучения (используется Whisper и Silero TTS, проверка CUDA)
import subprocess # Для запуска внешних процессов и команд (запуск приложений, 'where')
import winshell # Для работы со специфичными для Windows функциями оболочки (поиск папок, ярлыков .lnk)
from pathlib import Path # Для удобной и объектно-ориентированной работы с путями к файлам и папкам
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # Компоненты из pycaw для управления системной громкостью Windows
from thefuzz import process as fuzzy_process # Для нечеткого (приблизительного) сравнения строк (поиск приложений по названию)
from transliterate import translit, exceptions as translit_exceptions # Для транслитерации текста (преобразование кириллицы в латиницу для поиска)

# --- Константы и Настройки ---

# Получаем путь к папке, где лежит этот скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app_index = {} # Словарь для найденных приложений: { "имя_lowercase": "путь" }
volume_control = None # для хранения объекта управления громкостью

# --- НАСТРОЙКИ WHISPER ---
# Модели: tiny, base, small, medium, large (чем больше, тем точнее и медленнее)
# Для русского языка 'small' или 'medium' обычно хороший баланс. На CPU лучше 'tiny' или 'base'.
WHISPER_MODEL_NAME = "small"
# Автоматически использовать GPU (CUDA), если доступно
USE_GPU_IF_AVAILABLE = True
# ------------------------


# --- НАСТРОЙКИ SILERO TTS ---
SILERO_LANG = 'ru'
SILERO_MODEL_ID = 'v4_ru' # v3_1_ru или v4_ru - современные модели
SILERO_SPEAKER = 'eugene' # Мужской голос (другие варианты: aidar, baya)
# Частота дискретизации для выбранной модели! v4_ru обычно 48000 Hz, v3_1_ru может быть 24000 Hz
SILERO_SAMPLE_RATE = 48000
# --------------------------

# Ключ доступа Porcupine
PORCUPINE_ACCESS_KEY = "YOUR_ACCESS_KEY" #свой ключ
# Путь к файлу ключевого слова
PORCUPINE_KEYWORD_PATH = os.path.join(BASE_DIR, "triggerword", "Jarvis_en_windows_v3_0_0.ppn")
PORCUPINE_SENSITIVITY = 0.7 # Чувствительность детектора (0.0 - 1.0)

# Параметры аудиопотока
CHUNK = 512 # Размер буфера для Porcupine
RATE = 16000 # Частота дискретизации (Whisper предпочитает 16000 Hz)
FORMAT = pyaudio.paInt16 # Формат аудио
CHANNELS = 1 # Количество каналов

# Конфигурация Hugging Face API
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"
HUGGINGFACE_HEADERS = {"Authorization": "Bearer YOUR_ACCESS_TOREN"}

# Пороги для нечеткого поиска приложений (можно настраивать)
FUZZY_SCORE_CUTOFF_TRANSLIT = 75 # Порог для транслитерированного поиска
FUZZY_SCORE_CUTOFF_ORIGINAL = 70 # Порог для поиска по оригинальному русскому запросу

# Флаг активации и время последней команды
is_listening_for_command = False # Переменная-флаг, показывающая, ожидает ли ассистент сейчас команду (True) или только ключевое слово (False).
last_command_time = 0            # Хранит временную метку (timestamp) момента, когда была обработана последняя команда.

# --- Инициализация ---

# --- ОПРЕДЕЛЕНИЕ УСТРОЙСТВА (CPU/GPU) ---
device = None

if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
    device = "cuda"
    print(f"Обнаружен CUDA GPU, будет использоваться устройство: '{device}'.")
else:
    device = "cpu"
    print(f"CUDA GPU не найден или отключен, будет использоваться устройство: '{device}'.")
# ------------------------------------------

# --- ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА УПРАВЛЕНИЯ ГРОМКОСТЬЮ (COM (Component Object Model)) ---
def initialize_volume_control():
    """Инициализирует глобальный объект для управления громкостью."""
    global volume_control
    try:
        print("Инициализация интерфейса управления громкостью...")
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        # Сразу получаем конечный объект volume
        volume_control = cast(interface, POINTER(IAudioEndpointVolume))
        print("Интерфейс управления громкостью успешно инициализирован.")
        return True
    except Exception as e:
        print(f"!!! Ошибка инициализации управления громкостью: {e} !!!")
        volume_control = None # Сбрасываем в None в случае ошибки
        return False

# --- ИНИЦИАЛИЗАЦИЯ WHISPER ---
whisper_model = None
try:
    print(f"Загрузка модели Whisper '{WHISPER_MODEL_NAME}' на '{device}'...")
    if device == "cpu":
         print("Предупреждение: Распознавание Whisper на CPU может быть медленным.")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    print(f"Модель Whisper '{WHISPER_MODEL_NAME}' успешно загружена.")
except Exception as e:
    print(f"!!! Ошибка загрузки модели Whisper: {e} !!!")
    print("Убедитесь, что модель '{WHISPER_MODEL_NAME}' доступна и библиотека Whisper установлена корректно.")
    print("Также проверьте установку PyTorch и ffmpeg.")
    exit()
# -----------------------------

# --- ИНИЦИАЛИЗАЦИЯ TTS (Silero) ---
silero_model = None
print(f"Загрузка модели Silero TTS '{SILERO_MODEL_ID}' через torch.hub (может занять время)...")
try:
    # !!! Распаковываем кортеж !!!
    # torch.hub.load возвращает (модель, что-то_еще), нам нужна только модель [0]
    # Используем model, _ для игнорирования второго элемента
    silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=SILERO_LANG,
                                  speaker=SILERO_MODEL_ID)

    # Теперь silero_model - это сама модель, а не кортеж
    silero_model.to(device) # Перемещаем модель на CPU или GPU
    print(f"Модель Silero TTS '{SILERO_MODEL_ID}' успешно загружена на {device}.")

    # Дополнительно: Проверим, соответствует ли SAMPLE_RATE модели
    # (Может понадобиться global, если SILERO_SAMPLE_RATE используется где-то еще)
    if hasattr(silero_model, 'sample_rate') and silero_model.sample_rate != SILERO_SAMPLE_RATE:
         print(f"Предупреждение: Указанная SILERO_SAMPLE_RATE ({SILERO_SAMPLE_RATE}) "
               f"не совпадает с частотой модели ({silero_model.sample_rate}). "
               f"Будет использоваться частота модели: {silero_model.sample_rate}")
         SILERO_SAMPLE_RATE = silero_model.sample_rate # Обновляем частоту

except Exception as e:
    print(f"!!! Ошибка загрузки модели Silero TTS через torch.hub: {e} !!!")
    print("Убедитесь, что интернет работает (для скачивания модели/конфигурации),")
    print("и библиотеки silero, torch, torchaudio установлены корректно.")
# -----------------------------

# Инициализация Porcupine для Wake Word Detection
if not os.path.exists(PORCUPINE_KEYWORD_PATH):
     print(f"Ошибка: Файл ключевого слова Porcupine не найден: {PORCUPINE_KEYWORD_PATH}")
     exit()
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[PORCUPINE_KEYWORD_PATH],
        sensitivities=[PORCUPINE_SENSITIVITY]
    )
    print("Детектор ключевого слова Porcupine инициализирован.")
except pvporcupine.PorcupineError as pe:
    print(f"Ошибка инициализации Porcupine: {pe}")
    exit()
except Exception as e:
    print(f"Неизвестная ошибка при инициализации Porcupine: {e}")
    exit()


# --- Вспомогательные Функции ---

# --- Функция speak ---
def speak(text):
    """Произносит текст с помощью Silero TTS, преобразуя числа в слова."""
    global silero_model, device, SILERO_SPEAKER, SILERO_SAMPLE_RATE
    if not silero_model:
        print(f"Ассистент (TTS Модель не загружена): {text}")
        return
    if not text:
        print("Ассистент: (Получен пустой текст для озвучки)")
        return

    print(f"Ассистент: {text}") # Печатаем оригинальный текст

    try:
        # --- Преобразование чисел в слова ---
        processed_text = text
        # Находим все последовательности цифр в тексте
        numbers_in_text = re.findall(r'\d+', processed_text)
        if numbers_in_text:
            print(f"  [TTS Preprocessing] Найдены числа: {numbers_in_text}")
            for number_str in numbers_in_text:
                try:
                    # Конвертируем числовую строку в слова на русском
                    number_words = num2words(int(number_str), lang='ru')
                    # Заменяем первое вхождение числа на слова (чтобы не заменить часть другого числа)
                    processed_text = processed_text.replace(number_str, number_words, 1)
                    print(f"      '{number_str}' -> '{number_words}'")
                except ValueError:
                    print(f"      Не удалось конвертировать '{number_str}' в число.")
                except Exception as e_num:
                    print(f"      Ошибка конвертации '{number_str}': {e_num}")
            print(f"  [TTS Preprocessing] Текст после конвертации: {processed_text}")
        # --- Конец преобразования ---

        # Используем обработанный текст для синтеза
        with torch.no_grad():
            audio_tensor = silero_model.apply_tts(text=processed_text, # <--- Используем processed_text
                                                speaker=SILERO_SPEAKER,
                                                sample_rate=SILERO_SAMPLE_RATE,
                                                put_accent=True,
                                                put_yo=True)

        audio_numpy = audio_tensor.cpu().numpy()

        # --- Воспроизведение с помощью PyAudio ---
        p_tts = pyaudio.PyAudio()
        stream_tts = None
        try:
            stream_tts = p_tts.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=SILERO_SAMPLE_RATE,
                                    output=True)
            stream_tts.write(audio_numpy.tobytes())
        finally:
            if stream_tts:
                stream_tts.stop_stream()
                stream_tts.close()
            p_tts.terminate()
        # --- Конец воспроизведения ---

    except Exception as e:
        print(f"!!! Ошибка во время синтеза или воспроизведения речи Silero: {e} !!!")

# --- КОНЕЦ ФУНКЦИИ speak ---

def clean_response(response):
    """Обрезает текст до ближайшего завершенного предложения."""
    sentences = re.split(r'(?<=[.!?])\s+', response)  # Разделение по предложениям
    if sentences:
        return ' '.join(sentences[:-1]) if len(sentences) > 1 else sentences[0]
    return response

def ask_huggingface(prompt):
    """Отправляет запрос в Hugging Face и получает логически завершенный ответ."""
    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.3,  # Уменьшаем случайность
            "top_p": 0.9,        # Контролируем вероятностное распределение
            "max_new_tokens": 500
        }
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=HUGGINGFACE_HEADERS, json=data)

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]['generated_text'].strip()

            # Обрезаем ответ до завершенного предложения
            clean_text = clean_response(generated_text)
            return clean_text
        else:
            print(f"Ошибка запроса: {response.status_code}, {response.text}")
            return "Не удалось получить ответ от нейросети."
    except Exception as e:
        print(f"Ошибка подключения к Hugging Face: {e}")
        return "Ошибка при обращении к нейросети."


def record_command_with_timeout(stream, p, timeout_seconds=3, silence_threshold=500, max_silence_chunks=40):
    """Запись команды после триггера с динамическим продлением времени."""
    print("Запись команды...")
    speak("Слушаю вас")

    frames = []
    silence_count = 0
    min_chunks = int(RATE / CHUNK * timeout_seconds)

    for _ in range(min_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"Ошибка чтения из аудиопотока во время записи: {e}")
            return None

    while silence_count < max_silence_chunks:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if len(data) == CHUNK * pyaudio.PyAudio().get_sample_size(FORMAT):
                pcm_data = struct.unpack_from("h" * CHUNK, data)
                if max(pcm_data) > silence_threshold:
                    silence_count = 0
                else:
                    silence_count += 1
            else:
                 silence_count += 1
        except IOError as e:
            print(f"Ошибка чтения из аудиопотока во время продления записи: {e}")
            return None
        if not is_listening_for_command:
             print("Запись прервана.")
             return None

    print("Запись завершена.")
    audio_data = b''.join(frames)
    audio_filename = save_audio_to_file(audio_data)
    return audio_filename

def save_audio_to_file(audio_data, filename="command_temp.wav"):
    """Сохраняет записанное аудио в файл .wav"""
    filepath = os.path.join(BASE_DIR, filename)
    try:
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()
        return filepath
    except Exception as e:
        print(f"Ошибка сохранения аудио в файл {filepath}: {e}")
        return None

# --- ФУНКЦИЯ transcribe_audio ---
def transcribe_audio(audio_filename):
    """Распознавание речи из аудиофайла с помощью Whisper."""
    if not whisper_model:
        print("Ошибка: Модель Whisper не загружена.")
        return ""
    if not audio_filename or not os.path.exists(audio_filename):
        print("Ошибка: Аудиофайл для распознавания не найден.")
        return ""

    print(f"Распознавание аудиофайла '{audio_filename}' с помощью Whisper ({WHISPER_MODEL_NAME} на {device})...")
    start_time = time.time()

    try:
        result = whisper_model.transcribe(audio_filename, language="ru", fp16=False)
        recognized_text = result.get("text", "").strip()
        end_time = time.time()
        print(f"Распознано Whisper: '{recognized_text}' (за {end_time - start_time:.2f} сек)")
        return recognized_text


    except Exception as e:
        print(f"!!! Ошибка во время распознавания речи Whisper: {e} !!!")
        import traceback
        traceback.print_exc()

        print("Проверьте, установлен ли ffmpeg и добавлен ли он в системный PATH.")
        return ""
    finally:
        # Удаляем временный аудиофайл после распознавания
        try:
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
        except OSError as e:
            print(f"Не удалось удалить временный файл {audio_filename}: {e}")
# ------------------------------------------

# --- Функции для логирования команд в JSON ---
LOG_FILE_PATH = os.path.join(BASE_DIR, "command_log.json")

def load_command_log():
    """Загружает историю команд из JSON файла."""
    try:
        # Пробуем открыть файл на чтение
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            # Пробуем загрузить JSON, если файл пуст или некорректен, вернем []
            try:
                log_data = json.load(f)
                # Убедимся, что это список (на случай ручного изменения файла)
                return log_data if isinstance(log_data, list) else []
            except json.JSONDecodeError:
                return [] # Возвращаем пустой список при ошибке декодирования
    except FileNotFoundError:
        # Если файл не найден, возвращаем пустой список
        return []

def save_command_log(log_data):
    """Сохраняет историю команд в JSON файл."""
    try:
        # Открываем файл на запись, перезаписывая его содержимое
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
            # Записываем данные с отступом для читаемости, отключаем экранирование ASCII
            json.dump(log_data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        # Обрабатываем возможные ошибки записи
        print(f"Ошибка записи лога в файл {LOG_FILE_PATH}: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка при сохранении лога: {e}")

def log_command(command_text, result_text=None):
    """Добавляет запись о выполненной команде и её результате в лог."""
    if not command_text:
        return

    log_history = load_command_log()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "command": command_text,
        # Добавляем результат, если он передан
        "result": result_text if result_text is not None else "N/A"
    }
    log_history.append(log_entry)
    save_command_log(log_history)

# --- Конец функций для логирования ---

def get_system_volume():
    """Получение текущего уровня громкости на системе (0-100)."""
    global volume_control
    if not volume_control: # Проверка, что объект был инициализирован
        print("Ошибка: Интерфейс громкости не инициализирован.")
        return 0 # Возвращаем 0 или другое значение по умолчанию
    try:
        # Используем готовый объект volume_control
        current_volume_scalar = volume_control.GetMasterVolumeLevelScalar()
        return int(current_volume_scalar * 100)
    except Exception as e:
        print(f"Ошибка получения громкости: {e}")
        # Попытка переинициализации в случае сбоя (опционально)
        print("Попытка переинициализации управления громкостью...")
        if initialize_volume_control():
            try: # Пробуем еще раз
                current_volume_scalar = volume_control.GetMasterVolumeLevelScalar()
                return int(current_volume_scalar * 100)
            except Exception as e2:
                 print(f"Повторная ошибка получения громкости: {e2}")
        return 0 # Возвращаем 0, если все плохо


def set_volume(level):
    """Устанавливает громкость на заданный уровень (0-100%)."""
    global volume_control
    if not volume_control: # Проверка
        print("Ошибка: Интерфейс громкости не инициализирован.")
        speak("Не могу изменить громкость, контроллер не работает.")
        return

    level = max(0, min(100, level))
    try:
        # Используем готовый объект volume_control
        volume_control.SetMasterVolumeLevelScalar(level / 100.0, None)

    except Exception as e:
        print(f"Ошибка при установке громкости: {e}")
        speak("Не удалось изменить громкость")
        # Попытка переинициализации (опционально)
        print("Попытка переинициализации управления громкостью...")
        initialize_volume_control() # Пробуем восстановить

def adjust_volume(change):
    """Регулирует громкость на указанное значение."""
    # Получаем текущую громкость
    current_volume = get_system_volume()
    # Вычисляем новую, ограничиваем диапазон
    new_volume = max(0, min(100, current_volume + change))
    set_volume(new_volume)


# --- Функции для поиска и запуска приложений ---

CSIDL_COMMON_DESKTOPDIRECTORY = 0x0019  # Идентификатор папки "Общий рабочий стол"
SHGFP_TYPE_CURRENT = 0                 # Флаг для получения текущего пути

def get_common_desktop_path_ctypes():
    """
    Пытается получить путь к общему рабочему столу через ctypes и Windows API.
    Возвращает Path объект или None в случае неудачи.
    """
    print("    Пытаюсь получить путь к Common Desktop через ctypes...")
    buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    try:
        # Получаем указатель на функцию SHGetFolderPathW из shell32.dll
        shell32 = ctypes.windll.shell32
        hresult = shell32.SHGetFolderPathW(
            None,                          # hwndOwner (не используется)
            CSIDL_COMMON_DESKTOPDIRECTORY, # nFolder (ID папки)
            None,                          # hToken (не используется)
            SHGFP_TYPE_CURRENT,            # dwFlags
            buf                            # pszPath (буфер для результата)
        )
        if hresult == 0:
            path_str = buf.value
            if path_str and os.path.exists(path_str) and os.path.isdir(path_str):
                print(f"      [ctypes OK] Путь получен: {path_str}")
                return Path(path_str)
            else:
                 print(f"      [ctypes Info] API вернул путь, но он не существует или не папка: '{path_str}'")
                 return None
        else:
            print(f"      [ctypes Ошибка] SHGetFolderPathW вернул ошибку HRESULT: {hresult:#010x}")
            return None
    except Exception as e:
        print(f"      [ctypes Исключение] Ошибка при вызове Windows API: {type(e).__name__} - {e}")
        return None

def get_common_desktop_path_fallback():
    """
    Пытается определить путь к общему рабочему столу через переменную окружения PUBLIC.
    Возвращает Path объект или None.
    """
    print("    Пытаюсь определить путь к Common Desktop через переменную PUBLIC...")
    public_path_str = os.environ.get('PUBLIC')
    if not public_path_str:
        print("      [Fallback Ошибка] Переменная окружения PUBLIC не найдена.")
        public_path_str = 'C:\\Users\\Public' # Абсолютный крайний случай
        print(f"      [Fallback Info] Использую стандартный путь: {public_path_str}")

    try:
        assumed_path = Path(public_path_str) / 'Desktop'
        if assumed_path.exists() and assumed_path.is_dir():
             print(f"      [Fallback OK] Найден предполагаемый путь: {assumed_path}")
             return assumed_path
        else:
             print(f"      [Fallback Info] Предполагаемый путь не существует или не папка: {assumed_path}")
             return None
    except Exception as e:
        print(f"      [Fallback Исключение] Ошибка при проверке пути: {e}")
        return None


def build_app_index():
    """
    Сканирует папки для поиска приложений с обходом winshell для Common Desktop.
    """
    global app_index
    print("Индексирую приложения (с обходом winshell для Common Desktop)...")
    start_time = time.time()
    indexed_count = 0
    app_index = {}
    potential_paths_set = set()

    # --- Получение путей к стандартным папкам ---
    folder_identifiers = {
        "Start Menu (User)": ['startmenu', 'programs'],
        "Start Menu (Common)": ['common_startmenu', 'common_programs'],
        "Desktop (User)": ['desktop'],
    }

    print("Определение путей к стандартным папкам (кроме Common Desktop):")
    for name, ids in folder_identifiers.items():
        found_path_for_folder = None
        for folder_id in ids:
            try:
                path_str = winshell.folder(folder_id)
                if path_str and os.path.exists(path_str) and os.path.isdir(path_str):
                    found_path_for_folder = Path(path_str)
                    print(f"  [OK] {name} (id: '{folder_id}'): {found_path_for_folder}")
                    potential_paths_set.add(found_path_for_folder)
                    break
                else:
                    print(f"  [Info] {name} (id: '{folder_id}'): Путь не найден или не существует/не папка ('{path_str}')")
            except (ValueError, AttributeError, OSError, Exception) as e:
                print(f"  [Ошибка] {name} (id: '{folder_id}'): Не удалось получить путь - {type(e).__name__}: {e}")
        if not found_path_for_folder:
             print(f"  [Предупреждение] Не удалось найти действительный путь для {name}, используя ID: {ids}")

    # --- Обработка Common Desktop отдельно ---
    print("Определение пути к Common Desktop:")
    common_desktop_path = None
    try:
        # Сначала пробуем стандартный winshell ID
        path_str_ws = winshell.folder('common_desktop')
        if path_str_ws and os.path.exists(path_str_ws) and os.path.isdir(path_str_ws):
            common_desktop_path = Path(path_str_ws)
            print(f"  [OK] Desktop (Common) (id: 'common_desktop' через winshell): {common_desktop_path}")
        else:
             print(f"  [Info] Desktop (Common) (id: 'common_desktop' через winshell): Путь не найден или не существует/не папка ('{path_str_ws}')")
             # Если winshell не сработал, пробуем ctypes
             common_desktop_path = get_common_desktop_path_ctypes()
             if not common_desktop_path:
                  # Если и ctypes не сработал, пробуем fallback
                  common_desktop_path = get_common_desktop_path_fallback()

    except (ValueError, AttributeError, OSError, Exception) as e_ws:
        print(f"  [Ошибка] Desktop (Common) (id: 'common_desktop' через winshell): Не удалось получить путь - {type(e_ws).__name__}: {e_ws}")
        # Если winshell вызвал ошибку, сразу пробуем ctypes
        common_desktop_path = get_common_desktop_path_ctypes()
        if not common_desktop_path:
            # Если и ctypes не сработал, пробуем fallback
            common_desktop_path = get_common_desktop_path_fallback()

    if common_desktop_path:
        potential_paths_set.add(common_desktop_path)
    else:
        print(f"  [Предупреждение] Не удалось найти действительный путь для Desktop (Common) ни одним из методов.")

    # --- Добавление Program Files ---
    print("Определение путей к Program Files...")
    pf_env_vars = ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"]
    for pf_var in pf_env_vars:
         pf_path_str = os.environ.get(pf_var)
         if pf_path_str and os.path.exists(pf_path_str) and os.path.isdir(pf_path_str):
             potential_paths_set.add(Path(pf_path_str))
             print(f"  [OK] {pf_var}: {pf_path_str}")

    # --- Подготовка к сканированию ---
    search_paths = list(potential_paths_set)
    print(f"Итоговые папки для сканирования ({len(search_paths)} шт.): {[str(p) for p in search_paths]}")
    if not search_paths:
        print("Предупреждение: Не найдено папок для индексации. Индекс может быть неполным.")

    # --- Сканирование найденных папок ---
    print("Начинаю сканирование найденных папок...")
    max_depth = 4
    for search_path in search_paths:
        print(f"  Сканирую: {search_path} ...")
        is_program_files = any(pf_var in str(search_path) for pf_var in pf_env_vars if os.environ.get(pf_var))
        try:
            for root, dirs, files in os.walk(search_path, topdown=True):
                 current_depth = str(Path(root)).count(os.sep) - str(search_path).count(os.sep)
                 if is_program_files and current_depth >= max_depth:
                      dirs[:] = []
                      continue
                 for filename in files:
                     file_path = Path(root) / filename
                     file_lower = filename.lower()
                     if file_lower.endswith('.lnk'):
                         try:
                             if not file_path.is_file(): continue
                             shortcut = winshell.shortcut(str(file_path))
                             target_path = shortcut.path
                             if target_path and target_path.lower().endswith('.exe'):
                                 target_path_obj = Path(target_path)
                                 if target_path_obj.is_file():
                                     app_name = file_path.stem.lower()
                                     if app_name not in app_index:
                                         app_index[app_name] = str(target_path_obj.resolve())
                                         indexed_count += 1
                         except FileNotFoundError: pass
                         except Exception: pass
        except PermissionError: print(f"    Нет доступа к части {search_path}, пропускаю недоступные подпапки.")
        except Exception as e_scan: print(f"    Ошибка сканирования {search_path}: {type(e_scan).__name__} - {e_scan}")

    # --- Добавление стандартных приложений Windows ---
    print("Добавление стандартных приложений Windows...")
    common_apps = { "блокнот": "notepad.exe", "калькулятор": "calc.exe", "paint": "mspaint.exe", "проводник": "explorer.exe", "командная строка": "cmd.exe", "диспетчер задач": "taskmgr.exe", "панель управления": "control.exe", "wordpad": "wordpad.exe", "записки": "StikyNot.exe"}
    added_common = 0
    for name, path_or_cmd in common_apps.items():
         if name not in app_index and name.replace(" ", "") not in app_index:
             full_path = None
             if '.exe' in path_or_cmd:
                 try:
                     result = subprocess.run(['where', path_or_cmd], capture_output=True, text=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                     if result.returncode == 0 and result.stdout: full_path = result.stdout.splitlines()[0].strip()
                 except Exception: pass
             app_index[name] = full_path if full_path else path_or_cmd
             added_common += 1
    print(f"Добавлено стандартных приложений: {added_common}")

    # --- Завершение ---
    end_time = time.time()
    print(f"Индексация завершена за {end_time - start_time:.2f} сек.")
    print(f"Всего найдено уникальных приложений/ярлыков для запуска: {len(app_index)}")

    # --- !!! Вывод всего найденного индекса !!! ---
    print("\n" + "=" * 25 + " Найденные приложения в индексе " + "=" * 25)
    if app_index:
        # Сортируем по ключу (имени приложения) для удобства чтения
        sorted_app_items = sorted(app_index.items())
        for name, path in sorted_app_items:
            # Печатаем ключ (имя) и значение (путь)
            print(f"  Ключ: '{name}'  -->  Путь: '{path}'")
    else:
        print("  Индекс приложений пуст.")
    print("=" * (50 + len(" Найденные приложения в индексе ")))
    # --- Конец вывода всего индекса ---


number_words_map = { # Словарь для нормализации чисел
    "ноль": "0", "один": "1", "два": "2", "три": "3", "четыре": "4",
    "пять": "5", "шесть": "6", "семь": "7", "восемь": "8", "девять": "9",
}

def find_application_path_auto(query):
    global app_index
    if not app_index:
        print("Индекс приложений пуст или не был построен.")
        return None, None

    query_lower = query.lower().strip()
    print(f"Поиск приложения для запроса: '{query_lower}'")

    # --- Нормализация чисел ---
    normalized_query = query_lower
    for word, digit in number_words_map.items():
        normalized_query = re.sub(r'\b' + re.escape(word) + r'\b', digit, normalized_query)
    if normalized_query != query_lower:
        print(f"  Нормализация чисел -> '{normalized_query}'")
    # --- Конец нормализации ---

    # Используем normalized_query дальше
    query_translit = ""
    try:
        query_translit = translit(normalized_query, 'ru', reversed=True)
        print(f"  Транслитерация нормализованного -> '{query_translit}'")
    except translit_exceptions.LanguageDetectionError:
        print(f"  Не удалось определить язык для транслитерации запроса: '{normalized_query}'.")
        query_translit = normalized_query
    except Exception as e:
        print(f"  Ошибка транслитерации: {e}")
        query_translit = normalized_query

    best_match_translit = None
    if query_translit != normalized_query:
        best_match_translit = fuzzy_process.extractOne(
            query_translit, app_index.keys(), score_cutoff=FUZZY_SCORE_CUTOFF_TRANSLIT
        )

    best_match_original = fuzzy_process.extractOne(
        normalized_query, app_index.keys(), score_cutoff=FUZZY_SCORE_CUTOFF_ORIGINAL
    )

    final_match = None
    best_score = -1
    source = "" # Для отслеживания, какой метод дал лучший результат

    # Проверяем результат поиска по транслитерированному запросу
    if best_match_translit and best_match_translit[1] > best_score:
        final_match = best_match_translit
        best_score = final_match[1]
        source = "транслитерация нормализованного"

    # Проверяем результат поиска по оригинальному (нормализованному) запросу
    if best_match_original and best_match_original[1] > best_score:
        # Если оценка совпадения по оригиналу выше текущей лучшей оценки,
        # обновляем результат
        final_match = best_match_original
        best_score = final_match[1]
        source = "оригинал нормализованный"

    # Возвращение результата
    if final_match:
        # Если найдено совпадение любым из методов выше порога
        matched_name = final_match[0] # Имя ключа из app_index, которое совпало
        score = final_match[1]        # Процент совпадения
        found_path = app_index[matched_name] # Получаем путь из индекса по ключу
        # Выводим информативное сообщение, используя исходный запрос пользователя
        print(f"  Найдено совпадение для '{query_lower}' (через {source}): '{matched_name}' (оценка: {score}%) -> {found_path}")
        # Возвращаем имя из индекса (для озвучки) и путь к файлу
        return matched_name, found_path
    else:
        # Если совпадений выше порога не найдено
        print(f"  Не найдено подходящего приложения для запроса '{query_lower}' (и после нормализации) в индексе.")
        return None, None # Возвращаем None для имени и пути

# --- Функция launch_application ---
def launch_application(path, name):
    """Запускает приложение по указанному пути, используя os.startfile."""
    try:
        print(f"Запускаю: {name} ({path})")
        os.startfile(path)
        speak(f"Запускаю")
        return True
    except FileNotFoundError:
        speak(f"Не удалось найти файл для {name}")
        print(f"Ошибка: Файл не найден по пути {path}")
        return False
    except Exception as e:
        speak(f"Не получилось запустить {name}")
        print(f"Ошибка при запуске {path}: {e}")
        return False

def correct_command_fuzzy(input_command):
    """Поправляет команду по нечеткому совпадению с известными шаблонами."""
    known_commands = [
        "сделай громче", "сделай тише",
        "звук на",
        "найди", "поищи",
        "скажи", "расскажи",
        "время", "сколько времени",
        "спасибо", "благодарю",
        "пока", "до свидания", "стоп",
        "запусти", "открой", "включи", "старт", "запустить", "открыть", "включить"
    ]
    input_command = input_command.lower().strip()
    best_match = fuzzy_process.extractOne(input_command, known_commands, score_cutoff=80)
    if best_match:
        matched_prefix = best_match[0]
        print(f"Поправлено: '{input_command}' -> '{matched_prefix}'")
        if input_command.startswith(matched_prefix):
            return matched_prefix + input_command[len(matched_prefix):]
        else:
            return matched_prefix
    return input_command

# --- Основная логика команд ---
def execute_command(command):
    """Выполнение команды после распознавания."""
    global is_listening_for_command
    command = correct_command_fuzzy(command)
    print(f"Обработка команды: '{command}'")

    if not command:
        print("Пустая команда, игнорирую.")
        return

        # --- Команды Запуска Приложений ---
    launch_triggers = ["запусти", "открой", "включи", "старт", "запустить", "открыть", "включить"]
    triggered_launch = False
    app_name_query = ""

    for trigger in launch_triggers:
        # Проверяем, начинается ли команда с триггера и есть ли после него пробел или команда состоит только из триггера
        if command.startswith(trigger) and (len(command) == len(trigger) or command[len(trigger)] == ' '):
            # Извлекаем имя приложения после триггера
            app_name_query = command[len(trigger):].strip()
            if app_name_query: # Убеждаемся, что имя приложения не пустое
                triggered_launch = True
                break # Нашли триггер и имя, выходим из цикла

    if triggered_launch:
        response_text = None # Переменная для хранения текста ответа для лога
        print(f"Ищу приложение (авто): '{app_name_query}'")
        matched_name, app_path = find_application_path_auto(app_name_query) # Ищем приложение в индексе

        if app_path:
            # Если нашли приложение в индексе, пытаемся его запустить
            success = launch_application(app_path, matched_name)
            # Формируем текст ответа для лога на основе успеха запуска
            response_text = f"Запускаю {matched_name}" if success else f"Не получилось запустить {matched_name}"
        else:
            # Если не нашли в индексе, пробуем запустить через систему (командой start)
            try:
                print(f"Автопоиск не нашел, пробую запустить '{app_name_query}' через систему...")
                # Используем Popen, чтобы не блокировать основной поток, если приложение долго запускается
                subprocess.Popen(f'start "" "{app_name_query}"', shell=True)
                response_text = f"Пытаюсь запустить {app_name_query} через систему"
                speak(response_text) # Озвучиваем попытку
            except Exception as e:
                # Если системный запуск не удался
                print(f"Системный запуск '{app_name_query}' не удался: {e}")
                response_text = f"Не удалось найти или запустить {app_name_query}"
                speak(response_text) # Озвучиваем ошибку

        log_command(command, response_text) # Логируем команду и результат/ответ
        return

   # --- Команды управления громкостью ---
    elif "сделай громче" in command:
        adjust_volume(10)
        log_command(command, f"Громкость увеличена (тек. {get_system_volume()}%)")
    elif "сделай тише" in command:
        adjust_volume(-10)
        log_command(command, f"Громкость уменьшена (тек. {get_system_volume()}%)")
    elif "звук на" in command:
        response_text = None
        match = re.search(r"(звук на)\s*(\d+)", command)
        if match:
            try:
                level = int(match.group(2))
                set_volume(level)
                response_text = f"Громкость установлена на {level} процентов"
                speak(response_text)
            except ValueError:
                response_text = "Не удалось преобразовать уровень громкости в число."
                speak(response_text)
            except Exception as e:
                response_text = "Произошла ошибка при установке громкости."
                speak(response_text)
        else:
            response_text = "Не могу определить уровень громкости."
            speak(response_text)
        log_command(command, response_text)
        
    # --- Конец команд управления громкостью ---

    # --- Команда поиска в интернете ---
    elif command.startswith("найди") or command.startswith("поищи"):
        response_text = None # Переменная для хранения ответа для лога
        search_query = "" # Переменная для поискового запроса

        # Извлекаем поисковый запрос после триггерной фразы
        if command.startswith("найди"):
            # Берем все после "найди"
            search_query = command[len("найди"):].strip()
        elif command.startswith("поищи"):
            # Берем все после "поищи "
            search_query = command[len("поищи"):].strip()

        if search_query:
            # Если поисковый запрос не пустой
            print(f"Ищу в браузере: {search_query}")
            # Формируем URL для поиска в Google, заменяя пробелы на '+'
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            try:
                # Открываем URL в браузере по умолчанию
                webbrowser.open(search_url)
                response_text = f"Ищу в интернете: {search_query}"
                speak(response_text) # Озвучиваем действие
            except Exception as e:
                # Обработка возможных ошибок при открытии браузера
                print(f"Ошибка при открытии браузера: {e}")
                response_text = f"Не удалось открыть браузер для поиска: {search_query}"
                speak(response_text) # Озвучиваем ошибку
        else:
            # Если поисковый запрос пустой (например, сказали просто "поищи")
            response_text = "Что нужно найти?"
            speak(response_text) # Спрашиваем, что искать

        log_command(command, response_text)


    # --- Команда обращения к нейросети (Hugging Face) ---
    elif command.startswith("скажи") or command.startswith("расскажи"):
        response_text = None # Переменная для хранения ответа для лога
        user_request = ""    # Переменная для запроса пользователя

        # Извлекаем запрос пользователя после триггерного слова
        if command.startswith("скажи"):
            # Берем все после "скажи "
            user_request = command[len("скажи"):].strip()
        elif command.startswith("расскажи"):
            # Берем все после "расскажи "
            user_request = command[len("расскажи"):].strip()

        if user_request:
            # Если запрос не пустой
            # Проверяем, доступна ли функция для обращения к нейросети
            if 'ask_huggingface' in globals() and callable(ask_huggingface):
                print(f"Отправляю запрос в нейросеть: {user_request}")
                # Вызываем функцию для получения ответа от Hugging Face
                ai_response = ask_huggingface(user_request)
                response_text = ai_response # Сам ответ нейросети идет в лог
                speak(response_text) # Озвучиваем ответ
            else:
                # Если функция ask_huggingface не найдена или не является функцией
                response_text = "Функция ответа на вопросы не настроена или неисправна."
                print(response_text) # Выводим в консоль, так как speak может быть недоступен
                speak(response_text) # Пытаемся озвучить
        else:
            # Если запрос пустой (например, сказали просто "скажи")
            response_text = "Что вы хотите узнать?"
            speak(response_text) # Спрашиваем, что нужно сказать/рассказать

        log_command(command, response_text)

    # --- Другие команды ---
    elif "время" in command or "сколько времени" in command:
        response_text = f"Сейчас {time.strftime('%H:%M', time.localtime())}"
        speak(response_text)
        log_command(command, response_text)
        
    elif "спасибо" in command or "благодарю" in command:
        response_text = "Всегда пожалуйста!"
        speak(response_text)
        log_command(command, response_text)

    elif "пока" in command or "до свидания" in command or "стоп" in command:
        response_text = "До свидания!"
        speak(response_text)
        is_listening_for_command = False
        log_command(command, response_text)

    # --- Если ни одна команда не подошла ---
    else:
        response_text = "Извините, я не понял эту команду."
        speak(response_text)
        log_command(command, response_text) # Логируем здесь


# --- Главный цикл прослушивания ---
def continuous_listen():
    """Непрерывное прослушивание с детекцией Wake Word и распознаванием команд Whisper."""
    global is_listening_for_command, last_command_time
    p = None
    stream = None
    run_main_loop = True

    try:
        p = pyaudio.PyAudio()

        print("Доступные устройства ввода:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                 try:
                      name = dev_info['name'].encode('cp1252').decode('cp1251')
                 except Exception:
                      name = dev_info['name']
                 print(f"  {i}: {name} (Каналов: {dev_info['maxInputChannels']}, Частота: {dev_info['defaultSampleRate']})")

        default_input_device_index = p.get_default_input_device_info()['index']
        try:
             default_dev_name = p.get_device_info_by_index(default_input_device_index)['name'].encode('cp1252').decode('cp1251')
        except Exception:
             default_dev_name = p.get_device_info_by_index(default_input_device_index)['name']
        print(f"\nИспользуется устройство ввода по умолчанию: {default_input_device_index} - {default_dev_name}")


        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=default_input_device_index)

        print(f"\nГолосовой помощник запущен! Частота {RATE} Гц, Буфер {CHUNK} сэмплов.")
        speak(f"Ассистент готов.")

        while run_main_loop:
            try:
                pcm_data = stream.read(CHUNK, exception_on_overflow=False)
                if len(pcm_data) == CHUNK * pyaudio.PyAudio().get_sample_size(FORMAT):
                    pcm_struct = struct.unpack_from("h" * CHUNK, pcm_data)
                else:
                    continue

                keyword_index = porcupine.process(pcm_struct)
                current_time = time.time()

                if keyword_index >= 0 and not is_listening_for_command:
                    if current_time - last_command_time > 2:
                        print(f"\n>>> Обнаружено ключевое слово '{Path(PORCUPINE_KEYWORD_PATH).stem}'! <<<")
                        is_listening_for_command = True
                    else:
                         pass

                elif is_listening_for_command:
                    # Запись команды
                    audio_filename = record_command_with_timeout(stream, p)

                    if audio_filename:
                        # Распознавание через Whisper
                        command_text = transcribe_audio(audio_filename)
                        if command_text:
                             execute_command(command_text)
                        else:
                             print("Команда не распознана (тишина или ошибка Whisper).")
                             pass
                    else:
                        print("Ошибка записи команды.")
                        speak("Возникла проблема с записью звука.")

                    is_listening_for_command = False
                    last_command_time = time.time()
                    print(f"\nОжидание команды '{Path(PORCUPINE_KEYWORD_PATH).stem}'...")

            except IOError as e:
                print(f"\n!!! Ошибка ввода/вывода аудиопотока: {e} !!!")
                print("Возможно, проблема с микрофоном. Попытка перезапуска через 5 секунд...")
                if stream is not None:
                    try:
                        if stream.is_active(): stream.stop_stream()
                        stream.close()
                    except Exception as e_close: print(f"Ошибка при закрытии потока: {e_close}")
                if p is not None:
                    try: p.terminate()
                    except Exception as e_term: print(f"Ошибка при завершении PyAudio: {e_term}")
                time.sleep(5)
                print("Перезапуск...")
                continuous_listen()
                return

            except KeyboardInterrupt:
                # Обработка Ctrl+C
                print("\nПолучен сигнал прерывания (Ctrl+C). Завершаю работу...")
                run_main_loop = False

            except Exception as e:
                # Отлов других непредвиденных ошибок
                print(f"\n!!! Произошла неожиданная ошибка в главном цикле: {e} !!!")
                import traceback
                traceback.print_exc() # Печать полного стека ошибки для диагностики
                print("Продолжение работы через 5 секунд...")
                time.sleep(5)

    finally:

        print("\nОсвобождение ресурсов...")
        if porcupine is not None:
            try:
                porcupine.delete()
                print("Ресурсы Porcupine освобождены.")
            except Exception as e:
                print(f"Ошибка при освобождении Porcupine: {e}")

        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                print("Аудиопоток закрыт.")
            except Exception as e:
                print(f"Ошибка при закрытии аудиопотока: {e}")

        if p is not None:
            try:
                p.terminate()
                print("PyAudio завершен.")
            except Exception as e:
                print(f"Ошибка при завершении PyAudio: {e}")

        print("Голосовой ассистент выключен.")

# --- Точка входа в программу ---
def main():
    """Главная функция запуска голосового ассистента."""
    print("\n--- Запуск основного процесса ассистента ---")
    # Инициализируем управление громкостью ПЕРЕД индексацией
    if not initialize_volume_control():
         speak("Внимание, не удалось инициализировать управление громкостью.")
         exit() # если без громкости работа не нужна
         
    # Строим индекс приложений один раз при запуске
    build_app_index()
    # Запускаем основной цикл прослушивания
    continuous_listen()

if __name__ == "__main__":
    # Проверка на запуск как основной скрипт
    main()