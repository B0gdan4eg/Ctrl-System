# CLAUDE.MD - Быстрая навигация по проекту Ctrl-System

> Документация для AI-ассистента Claude для быстрой ориентации в проекте

---

## 🎯 КРАТКОЕ ОПИСАНИЕ ПРОЕКТА

**Ctrl-System** - это десктопное приложение на Python для автоматической детекции дефектов на изображениях с использованием глубокого обучения (U-Net).

- **Тип**: Desktop Application (PySide6/Qt)
- **Язык**: Python 3.x
- **Назначение**: Контроль качества через анализ изображений
- **Архитектура**: Модульная MVC-подобная структура
- **Многопоточность**: Да (QThread workers)
- **Аутентификация**: НЕТ (single-user app)

---

## 📦 ТЕХНИЧЕСКИЙ СТЕК

```python
opencv-python >= 4.8.0      # Обработка изображений
numpy >= 1.24.0             # Численные вычисления
torch >= 2.0.0              # Deep Learning (U-Net)
PySide6 >= 6.5.0            # GUI Framework (Qt)
matplotlib >= 3.7.0         # Визуализация
```

**Важно**: Нет Flask/Django/FastAPI - это НЕ веб-приложение!

---

## 📁 СТРУКТУРА ПРОЕКТА

```
D:\Ctrl-System/
│
├── main.py                           # ⭐ ТОЧКА ВХОДА - запуск приложения
│
├── config/
│   └── settings.py                   # ⚙️ ВСЕ настройки приложения (параметры, пути, UI размеры)
│
├── models/
│   ├── __init__.py
│   ├── unet.py                       # 🧠 U-Net нейросеть для сегментации дефектов
│   └── zone_unet.pth                 # 🆕 Веса U-Net для сегментации зон (см. ниже)
│
├── core/                             # 🔧 ЯДРО - основная бизнес-логика
│   ├── __init__.py
│   ├── image_processing.py           # Обработка изображений (нормализация, фильтры)
│   ├── zone_detection.py             # Классический поиск зон (CLAHE+contours)
│   ├── zone_segmentation.py          # 🆕 Нейросетевая сегментация зон (ZoneSegmenter)
│   ├── defect_detection.py           # Детекция дефектов с помощью U-Net
│   └── cameras/                      # Драйверы камер
│       ├── camera.py                 # USB через OpenCV
│       ├── ebus_camera.py            # GigE через Pleora eBUS (JAI)
│       ├── pylon_camera.py           # GigE через Basler Pylon
│       └── gige_camera.py            # Общий GigE интерфейс
│
├── workers/                          # 🔄 МНОГОПОТОЧНОСТЬ
│   ├── __init__.py
│   ├── processing_worker.py          # Воркер для обработки изображений (QThread)
│   └── camera_worker.py              # Воркер для захвата с камеры (QThread)
│
├── ui/                               # 🎨 ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС
│   ├── __init__.py
│   ├── main_window.py                # ⭐ Главное окно приложения
│   │
│   ├── widgets/                      # Переиспользуемые UI компоненты
│   │   ├── __init__.py
│   │   ├── image_viewer.py           # Виджет просмотра изображений
│   │   ├── control_panel.py          # Панель управления параметрами
│   │   ├── stats_panel.py            # Панель статистики и логов
│   │   └── camera_widget.py          # Виджет для работы с камерой
│   │
│   └── styles/
│       └── dark_theme.py             # Темная тема оформления
│
├── utils/
│   ├── __init__.py
│   └── logger.py                     # 📝 Система логирования
│
├── scripts/                          # 🛠️ УТИЛИТЫ И ML PIPELINE
│   ├── TRAIN.py                      # Обучение U-Net для дефектов
│   ├── generate_zone_dataset.py      # 🆕 Авто-разметка датасета зон (weak supervision)
│   ├── train_zone_unet.py            # 🆕 Обучение U-Net для зон (GPU)
│   ├── predict_zones.py              # 🆕 Инференс зон + сравнение с классикой
│   ├── zone_detection_lab.py         # Лаборатория алгоритмов поиска зон
│   ├── defect_detection_pipeline.py  # Полный пайплайн обработки
│   ├── build_exe.py                  # Сборка .exe через PyInstaller
│   ├── cuting.py / hist.py / prep.py # Утилиты подготовки данных
│
├── addition/                         # Эксперименты разработки
│   └── zone_detection_dev.py         # Прототипы поиска зон
│
├── requirements.txt                  # Зависимости проекта
├── README.md                         # Основная документация
├── best_model.pth                    # Обученная модель U-Net
│
└── docs/
    └── CLAUDE.MD                     # ⭐ ЭТОТ ФАЙЛ - документация для Claude
```

---

## 🔑 КЛЮЧЕВЫЕ ФАЙЛЫ (что читать в первую очередь)

### 1. **main.py** - Точка входа
- Инициализация QApplication
- Создание главного окна `DefectDetectionApp`
- Запуск event loop
- Обработка критических ошибок

### 2. **config/settings.py** - Центр конфигурации
```python
# Параметры обработки (актуально на 13.05.2026)
DEFAULT_NUM_ZONES = 5                  # Количество зон для поиска
DEFAULT_MIN_AREA_RATIO = 0.3           # Мин. размер зоны (30% от изображения)
DEFAULT_MAX_AREA_RATIO = 0.95          # Макс. размер зоны (95%)
DEFAULT_STD_RANGE = 3.0                # std для histogram normalization (UI default)
DEFAULT_DETECTION_THRESHOLD = 0.01     # Порог детекции (низкий — для мелких дефектов)
DEFAULT_MODEL_SIZE = 256               # Размер тайла модели (256×256, sliding window)

# Пути
MODEL_PATH = "best_model.pth"          # Путь к обученной модели дефектов
ZONE_MODEL_PATH = "models/zone_unet.pth"  # Путь к U-Net сегментации зон
USE_AI_ZONE_DETECTION = True              # True = нейросеть, фолбэк — классика

# UI
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000

# Модель U-Net
MODEL_IN_CHANNELS = 3                  # RGB вход
MODEL_OUT_CHANNELS = 1                 # Бинарная маска выхода
MODEL_FEATURES = [64, 128, 256, 512]   # Архитектура encoder/decoder

# Предобработка
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75

# Визуализация
COLOR_DEFECT = (0, 0, 255)             # Красный для дефектов
COLOR_ZONE_BORDER = (0, 255, 0)        # Зеленый для границ зон
```

### 3. **ui/main_window.py** - Главное окно
- Наследуется от `QMainWindow`
- Создает вкладки (TabWidget): Оригинал / Результат / Камера
- Инициализирует все виджеты (ImageViewer, ControlPanel, StatsPanel)
- Подключает сигналы/слоты
- Запускает `ProcessingWorker` при обработке
- Обновляет UI по результатам обработки

### 4. **models/unet.py** - Нейросеть
- Класс `UNet` для сегментации
- Encoder-Decoder архитектура
- Skip connections между encoder/decoder
- Вход: (B, 3, H, W) - RGB изображение
- Выход: (B, 1, H, W) - маска дефектов

### 5. **core/defect_detection.py** - Основная логика
- Класс `DefectDetector` - главный обработчик
- Загрузка модели из `best_model.pth`
- Метод `process_image()` - основной пайплайн
- Детекция дефектов с использованием U-Net

---

## 🔄 ПАЙПЛАЙН ОБРАБОТКИ (как это работает)

```
1. Загрузка изображения
   ↓ (image_processing.load_image)

2. Поиск зон интереса
   ↓ (zone_detection.find_zones)
   - CLAHE enhancement
   - Поиск контуров
   - Фильтрация по площади

3. Для каждой зоны:
   ↓
   3a. Нормализация (опционально)
       ↓ (image_processing.normalize_zone)
       - Mean ± Std Dev нормализация

   3b. Предобработка
       ↓ (image_processing.preprocess_for_model)
       - Bilateral фильтр (шумоподавление)
       - Resize до DEFAULT_MODEL_SIZE (512x512)
       - Конвертация в тензор PyTorch

   3c. Детекция дефектов
       ↓ (defect_detection.detect_defects)
       - Forward pass через U-Net
       - Sigmoid активация
       - Бинаризация по порогу

   3d. Постобработка маски
       ↓
       - Resize обратно к размеру зоны
       - Наложение на оригинал

4. Визуализация результатов
   ↓ (defect_detection.visualize_results)
   - Наложение красных масок дефектов
   - Рисование зеленых границ зон
   - Вывод статистики
```

---

## ✨ УЛУЧШЕННАЯ ДЕТЕКЦИЯ ЗОН (НОВОЕ!)

### Добавлен улучшенный метод `find_top_zones_improved()`

**Новые возможности:**
- **Коррекция изображения**: Яркость (-100 до +100) и Контраст (0.1 до 5.0)
- **Фильтр по абсолютной площади**: Отсекает мелкий мусор (min_absolute_area)
- **Фильтр по пропорциям**: Отсекает слишком узкие/вытянутые объекты (max_aspect_ratio)
- **Фильтр по компактности**: Отсекает рваные контуры (circularity)
- **Фильтр по заполненности**: Отсекает Г-образные формы (extent)
- **Анализ зон**: Функция `analyze_zones()` для статистики

**Пример использования:**
```python
from core import find_top_zones_improved, analyze_zones, apply_image_correction

# С коррекцией изображения
contours = find_top_zones_improved(
    img_gray,
    top_n=5,
    min_area_ratio=0.9,           # 90% от максимальной зоны
    min_absolute_area=20000,      # Минимум 20000 пикселей
    max_aspect_ratio=20.0,        # Максимум 20:1
    brightness=100,               # +100 яркости
    contrast=5.0                  # x5 контраст
)

# Статистика зон
stats = analyze_zones(contours, img_gray.shape)
print(f"Найдено зон: {stats['count']}")
print(f"Средняя площадь: {stats['mean_area']:.0f}")
print(f"Покрытие изображения: {stats['total_coverage']:.1f}%")
```

**Параметры в config/settings.py:**
```python
DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 1.0
MIN_ABSOLUTE_AREA = 20000      # Из тестового UI
MAX_ASPECT_RATIO = 20.0        # Из тестового UI
```

---

## 🤖 НЕЙРОСЕТЕВАЯ СЕГМЕНТАЦИЯ ЗОН (v1.1, 2026-05-05)

Для решения корневой проблемы Задачи 5 (классика захватывала тёмную рамку
фильтра в маску зоны) добавлена **U-Net сегментация зон** через
`ZoneSegmenter`. Используется как основной метод; классика — фолбэк.

### Файлы

| Файл | Назначение |
|------|-----------|
| `core/zone_segmentation.py` | Класс `ZoneSegmenter` (lazy-load, GPU/CPU авто) |
| `models/zone_unet.pth` | Обученные веса (~31 MB, val IoU = 0.977) |
| `scripts/generate_zone_dataset.py` | Авто-разметка датасета (weak supervision) |
| `scripts/train_zone_unet.py` | Обучение U-Net для зон |
| `scripts/predict_zones.py` | Инференс + сравнение с классикой |
| `docs/ai_zone_segmentation.md` | Полный отчёт о реализации |

### API ZoneSegmenter

```python
from core.zone_segmentation import ZoneSegmenter

seg = ZoneSegmenter(model_path="models/zone_unet.pth")
# is_available — модель загрузилась? (lazy-check)
# predict_mask(image) — бинарная маска
# predict_contours(image, top_n=5) — drop-in замена find_top_zones()
```

### Где это включается

В `workers/processing_worker.py` есть singleton `_zone_segmenter` и логика:
1. Если `USE_AI_ZONE_DETECTION = True` и модель доступна → пытаемся U-Net
2. Если U-Net вернула пусто → авто-фолбэк на `find_top_zones()`
3. В лог пишется метод: `✓ Найдено зон: 5 (U-Net)` / `(классика)`

### Как переобучить

См. `docs/ai_zone_segmentation.md` — полный пайплайн. TL;DR:
```bash
# 1. Авто-разметка
python scripts/generate_zone_dataset.py --input-dir <корпус> \
    --output-dir <датасет> --top-n 5 --min-zones 4

# 2. Обучение (нужен глобальный Python 3.11 с GPU torch)
"C:/.../Python311/python.exe" scripts/train_zone_unet.py \
    --dataset <датасет> --output <out> --epochs 50 --batch-size 4

# 3. Замена в проекте
cp <out>/best_zone_unet.pth models/zone_unet.pth
```

### Важно для разработчика

- **Двойной Python:** проектный `venv311` имеет CPU-torch (для запуска приложения),
  глобальный `Python311` — GPU-torch (для тренировки)
- **Drop-in совместимость:** `predict_contours()` возвращает контуры в том же
  формате что `find_top_zones()` — никаких изменений в `_process_zone()` не нужно
- **Выключатель:** `USE_AI_ZONE_DETECTION = False` in `config/settings.py`
  возвращает приложение к классическому поведению

---

## 🔬 ПАЙПЛАЙН ДЕТЕКЦИИ ДЕФЕКТОВ (v1.2, 2026-05-13)

Pipeline инференса полностью переработан. Текущий артефакт:
`best_model.pth` (epoch 48, EMA val_iou=0.5145, обучен на red-zone датасете).

### 1. Tiled inference

`prepare_zone_tiles()` и `stitch_tile_probs()` в `core/image_processing.py`:
- ROI режется на тайлы **256×256** (sliding window, overlap 25%)
- Косинусное окно для weighted blending → бесшовное соединение
- Маленькие зоны (≤ tile_size) → один летербокс-тайл (как раньше)

### 2. Multi-contrast fusion

В `_prepare_zone()` (workers/processing_worker.py):
- Для каждой зоны делаются **2 варианта** histogram normalization (`std_range=2.0` и `3.0`)
- Каждый вариант проходит:
  - Sliding window 256×256 → probs map (stitched)
  - Letterbox 256×256 (вся зона в один тайл) → probs map (restored)
- 4 probs map'а на зону объединяются через `np.maximum.reduce` (любой режим увидел дефект → пиксель помечен)

### 3. Edge-aware refinement

`refine_defect_mask()` в `core/defect_detection.py`:
1. **Guided filter** (R=4, eps=0.005) на probs с grayscale guide — границы «защёлкиваются» на яркостных перепадах
2. **Hysteresis thresholding** (strong > 0.7, edge > 0.5, weak > threshold_low)
3. **Морфология** (open + close, 3×3 ellipse)
4. **`_intensity_shrink()`** — для каждой компоненты:
   - Строится адаптивное кольцо-фон (`ring_width = max(25, max_dim*0.6+8)`)
   - Safety check: при шумном фоне (`bg_std > 22`) — пропускается
   - Пиксели маски с яркостью близкой к фону (`|guide - bg_mean| < k_sigma * bg_std`) удаляются
   - k_sigma = 2.5 для малых компонент, 1.5 для больших (>1500 px) — мягче, чтобы не съесть низко-контрастные дефекты

### 4. Производительность

| Устройство | Время/кадр | Оптимизации |
|---|---|---|
| **GPU (RTX 5050)** | ~2 сек | `channels_last`, batch по 8 |
| **CPU** | ~14 сек | `set_num_threads`, MKL-DNN, `channels_last` |

`_apply_device_optimizations()` в `ui/main_window.py` срабатывает автоматически
после загрузки модели по `self.device.type`. `torch.compile` НЕ применяется на
Windows CPU (требует MSVC build tools).

### 5. Стандартные значения параметров

```python
DEFAULT_MODEL_SIZE = 256           # Размер тайла (sliding + letterbox)
DEFAULT_DETECTION_THRESHOLD = 0.01 # Низкий порог для мелких дефектов
DEFAULT_STD_RANGE = 3.0            # UI default (в коде же используются 2.0 + 3.0)
HIST_CONTRASTS = (2.0, 3.0)        # ProcessingWorker — multi-contrast fusion
```

---

## 📐 КЛАССИФИКАЦИЯ ДЕФЕКТОВ ПО ГОСТ 11141-84 (v1.2, 2026-05-13)

`classify_defects_gost(mask, pixel_size_mm)` в `core/defect_detection.py`:

### Алгоритм
1. Для каждой связной компоненты в маске:
   - `aspect_ratio = max(w,h) / min(w,h)`
   - Если `aspect > 3` → **царапина**, иначе → **точка** (по ГОСТ Приложение 2)
   - `size_mm = max(w, h) * PIXEL_SIZE_MM` (большая ось в мм)
   - Поиск минимального класса, в лимит которого укладывается размер

### Лимиты (мм)

| Класс | Точка | Царапина |
|---|---|---|
| I | ≤ 0.020 | ≤ 0.004 |
| II | ≤ 0.050 | ≤ 0.006 |
| III | ≤ 0.100 | ≤ 0.010 |
| **IV** | **≤ 0.300** | **≤ 0.020** |
| V | ≤ 0.500 | ≤ 0.040 |
| VI | ≤ 0.700 | ≤ 0.060 |
| VII | ≤ 1.000 | ≤ 0.100 |
| VIIIa | ≤ 1.500 | ≤ 0.300 |
| IX | ≤ 3.000 | ≤ 0.400 |
| хуже IXa | > 3.0 | > 0.4 |

### Вердикт
- **Класс детали = худший класс среди всех дефектов**
- I-IV → **ГОДЕН** (зелёный)
- V и хуже → **БРАК** (красный)
- Класс показывается всегда (даже когда ГОДЕН)

Возвращает dict:
```python
{
    'defects': [{'kind': 'point'|'scratch', 'size_mm': ..., 'bbox': (x,y,w,h),
                 'area_px': ..., 'class': 'IV'}, ...],
    'worst_class': 'IV',
    'count_points': N,
    'count_scratches': M,
}
```

---

## 🎨 UI КОМПОНЕНТЫ (ui/widgets/)

### ImageViewer (image_viewer.py)
- Наследуется от `QGraphicsView` (с 13.05.2026 — было `QLabel`)
- **Зум колесом мыши** (range 0.05× — 40×, по позиции курсора)
- **Pan через drag** левой кнопкой (`ScrollHandDrag`)
- **Double-click → reset zoom** (fit-to-view)
- Совместимый API: `display_image(np.ndarray)`, `clear_image()`
- При первом показе fit-to-view, после пользовательского зума масштаб сохраняется

### ControlPanel (control_panel.py)
- Панель с элементами управления параметрами обработки
- Спинбоксы для:
  - Количество зон
  - Порог детекции (default 0.01, step 0.01)
  - Std Range / контраст (default 3.0)
- **Чекбокс «Показывать выделение дефектов»** — toggle красной заливки без повторного прогона модели
- Кнопки: "Загрузить изображение", "Запустить обработку", "Сохранить результат"
- Сигнал `show_defects_changed(bool)` для toggle overlay

### StatsPanel (stats_panel.py)
- Панель статистики и логов
- `QTextEdit` для вывода логов в реальном времени
- Отображает:
  - Количество найденных зон
  - Количество дефектов
  - Размеры зон
  - Время обработки

### CameraWidget (camera_widget.py)
- Виджет для работы с камерой
- Захват кадров с USB/IP камеры
- Использует `CameraWorker` для асинхронного захвата

---

## 🔄 МНОГОПОТОЧНОСТЬ (workers/)

### ProcessingWorker (processing_worker.py)
- Наследуется от `QThread`
- Выполняет тяжелую обработку в фоновом потоке
- НЕ блокирует UI во время обработки
- **Сигналы:**
  - `progress_updated(int)` - прогресс 0-100%
  - `log_message(str)` - сообщения для лога
  - `processing_complete(dict)` - результаты обработки
  - `error_occurred(str)` - ошибки

**Использование:**
```python
worker = ProcessingWorker(detector, image, params)
worker.progress_updated.connect(self.update_progress)
worker.processing_complete.connect(self.on_complete)
worker.start()
```

### CameraWorker (camera_worker.py)
- Аналогично ProcessingWorker
- Захват кадров с камеры в фоновом режиме
- Эмитирует сигнал `frame_captured(np.ndarray)` для каждого кадра

---

## 📝 ЛОГИРОВАНИЕ (utils/logger.py)

```python
# Использование:
from utils import app_logger

app_logger.info("Информационное сообщение")
app_logger.warning("Предупреждение")
app_logger.error("Ошибка", exc_info=True)
app_logger.debug("Отладочная информация")
```

- **Формат логов**: `[YYYY-MM-DD HH:MM:SS] [LEVEL] message`
- **Файлы**: `logs/defect_detection_YYYYMMDD.log`
- **Одновременный вывод**:
  - В файл (rotation daily)
  - В консоль (stdout)
  - В UI (StatsPanel через custom handler)

---

## 🧠 МОДЕЛЬ U-NET (models/unet.py)

### Архитектура
```
Input: (B, 3, H, W) RGB image
  ↓
Encoder:
  Conv1: 3 → 64
  Conv2: 64 → 128
  Conv3: 128 → 256
  Conv4: 256 → 512

Bottleneck: 512 → 1024

Decoder (with skip connections):
  UpConv4: 1024 + 512 → 512
  UpConv3: 512 + 256 → 256
  UpConv2: 256 + 128 → 128
  UpConv1: 128 + 64 → 64
  ↓
Output: (B, 1, H, W) Binary mask
```

### Загрузка модели
```python
from models import UNet
from config.settings import MODEL_PATH, MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS

model = UNet(in_channels=MODEL_IN_CHANNELS, out_channels=MODEL_OUT_CHANNELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
```

---

## 🛠️ ВАЖНЫЕ МОМЕНТЫ ДЛЯ CLAUDE

### ❌ ЧТО НЕ НУЖНО ИСКАТЬ
- **Нет аутентификации/авторизации** - это single-user app
- **Нет базы данных** - работает с файлами напрямую
- **Нет REST API** - это не веб-приложение
- **Нет тестов** - пока не реализованы
- **Нет Docker** - запускается напрямую через Python

### ✅ ЧТО ВАЖНО ЗНАТЬ
1. **Обработка ВСЕГДА асинхронная** через QThread workers
2. **Все параметры в config/settings.py** - не хардкодятся
3. **Модель загружается один раз** при инициализации DetectorDetection
4. **Логи дублируются** в файл + консоль + UI
5. **UI обновляется через Qt signals/slots** - thread-safe коммуникация

### 🔍 КУДА СМОТРЕТЬ ПРИ РАЗНЫХ ЗАДАЧАХ

| Задача | Файлы |
|--------|-------|
| Изменить параметры обработки | `config/settings.py` |
| Добавить новый виджет UI | `ui/widgets/` + `ui/main_window.py` |
| Изменить логику детекции дефектов | `core/defect_detection.py` |
| Изменить предобработку | `core/image_processing.py` |
| Изменить классический поиск зон | `core/zone_detection.py` |
| Изменить нейросетевой поиск зон | `core/zone_segmentation.py` |
| Изменить архитектуру сети | `models/unet.py` |
| Переобучить модель зон | `scripts/train_zone_unet.py` |
| Добавить новый воркер | `workers/` |
| Изменить стиль UI | `ui/styles/dark_theme.py` |
| Проблемы с логами | `utils/logger.py` |
| Обучение модели дефектов | `scripts/TRAIN.py` |

### 🐛 ТИПИЧНЫЕ ПРОБЛЕМЫ

1. **"Model not found"**
   - Проверить наличие `best_model.pth` в корне проекта
   - Проверить `MODEL_PATH` в `config/settings.py`

2. **"Image not loading"**
   - Проверить поддерживаемые форматы (PNG, JPG, BMP через OpenCV)
   - Проверить кодировку пути (Windows paths)

3. **"UI freezes"**
   - Убедиться что обработка идет через ProcessingWorker
   - Не делать тяжелые операции в main thread

4. **"Zones not detected"**
   - Проверить параметры CLAHE в `zone_detection.py`
   - Проверить `MIN_AREA_RATIO` и `MAX_AREA_RATIO` в settings

---

## 🚀 БЫСТРЫЙ СТАРТ

### Запуск приложения
```bash
cd D:\Ctrl-System
python main.py
```

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Проверка модели
```bash
python -c "import torch; print(torch.load('best_model.pth').keys())"
```

---

## 📊 СТАТИСТИКА ПРОЕКТА

- **Количество модулей**: ~30 файлов Python
- **Основные зависимости**: 5 библиотек
- **Размер модели**: ~100-200 MB (U-Net checkpoint)
- **Поддерживаемые форматы изображений**: PNG, JPG, BMP, TIFF
- **Target платформа**: Windows (но кроссплатформенный код)

---

## 🎯 ФИЛОСОФИЯ КОДА

1. **Модульность** - каждый компонент независим
2. **Разделение ответственности**:
   - `core/` - бизнес-логика
   - `ui/` - интерфейс
   - `models/` - нейросети
   - `workers/` - многопоточность
   - `config/` - настройки
3. **Асинхронность** - никакой блокировки UI
4. **Переиспользуемость** - виджеты как отдельные компоненты
5. **Настраиваемость** - все в config/settings.py

---

## 📚 ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ

- **README.md** - подробная документация на русском
- **addition/TRAIN.py** - если нужно переобучить модель
- **zone_finder_improved.py** - standalone скрипт для экспериментов с зонами

---

## ⚡ QUICK REFERENCE

### Импорты
```python
# Основное приложение
from ui import DefectDetectionApp

# Детектор
from core.defect_detection import DefectDetector

# Модель
from models import UNet

# Настройки
from config import settings

# Логгер
from utils import app_logger

# Виджеты
from ui.widgets import ImageViewer, ControlPanel, StatsPanel, CameraWidget

# Воркеры
from workers import ProcessingWorker, CameraWorker
```

### Основной флоу кода
```python
# 1. Инициализация
app = QApplication(sys.argv)
window = DefectDetectionApp()

# 2. Загрузка изображения
image = cv2.imread(filepath)

# 3. Создание детектора
detector = DefectDetector(model_path=settings.MODEL_PATH)

# 4. Запуск обработки
worker = ProcessingWorker(detector, image, params)
worker.start()

# 5. Обработка результатов
@Slot(dict)
def on_complete(results):
    original = results['original']
    result_image = results['result']
    zones = results['zones']
    stats = results['stats']
```

---

**КОНЕЦ ДОКУМЕНТАЦИИ**

_Последнее обновление: 2026-05-13_
_Версия проекта: 1.2_
_Claude, ты готов! 🚀_
