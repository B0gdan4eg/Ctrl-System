# JAI Camera Viewer - eBUS SDK Edition

Простое приложение для автоматического обнаружения, подключения и просмотра видео с JAI GigE Vision камер через официальный eBUS SDK.

## Возможности

- Автоматическое обнаружение JAI камер в сети через eBUS SDK
- Автоматическое подключение к первой найденной камере
- Отображение live видео через OpenCV
- Сохранение кадров по нажатию 's'
- Простой интерфейс без меню
- Использование официального JAI/Pleora eBUS SDK

## Требования

### Системные требования
- **Python 3.11** (обязательно)
- **Windows 10/11** (64-bit)
- **eBUS SDK C++** (должен быть установлен в `C:\Program Files\JAI\eBUS SDK`)
- **JAI GigE Vision камера** (например, RM-4200GE)
- **Gigabit Ethernet** подключение

### Python зависимости
- `numpy>=1.21.0,<2.0` - Обработка массивов (ВАЖНО: версия 1.x для совместимости с eBUS)
- `opencv-python>=4.5.0,<4.10` - Отображение видео
- `ebus_python-6.5.4` - eBUS SDK Python API (устанавливается отдельно)

## Установка

### 1. Установить Python зависимости

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Установить eBUS SDK Python API

#### Вариант A: Wheel-файл в проекте

Файл уже находится в корне проекта:

```bash
py -3.11 -m pip install ebus_python-6.5.4-7277_jai-py311-none-win_amd64.whl
```

#### Вариант B: Скачать с сайта JAI

1. Перейти на https://www.jai.com/support-software/jai-software
2. Найти `ebus_python-6.5.4-7277_jai-py311-none-win_amd64.whl` (2.2 MB)
3. Скачать и установить:
```bash
py -3.11 -m pip install путь\к\скачанному\ebus_python-6.5.4-7277_jai-py311-none-win_amd64.whl
```

### 3. Откатить NumPy (важно!)

eBUS SDK требует NumPy версии 1.x:

```bash
py -3.11 -m pip install "numpy<2"
```

### 4. Проверить установку

```bash
py -3.11 -c "import eBUS; print('eBUS SDK installed successfully!')"
```

Если команда выполнилась без ошибок - всё готово!

## Запуск

### Простой запуск

```bash
py -3.11 main.py
```

Приложение автоматически:
1. Найдёт JAI камеру в сети (через eBUS SDK)
2. Подключится к первой найденной камере
3. Откроет окно с live видео

### Управление

В окне live видео:
- **q** или **ESC** - Выход из приложения
- **s** - Сохранить текущий кадр (файлы `frame_0000.png`, `frame_0001.png`, ...)

### Пример вывода

```
======================================================================
    JAI Camera Viewer - eBUS SDK Edition
    Automatic Discovery & Live Video Stream
======================================================================

Searching for JAI cameras...

Found 1 camera(s):
  [0] JAI RM-4200GE at 192.168.1.50
      S/N: 12345678, MAC: 00:11:22:33:44:55

Connecting to camera [0]...
Successfully connected!

Camera Information:
  manufacturer: JAI
  model: RM-4200GE
  serial: 12345678
  ip: 192.168.1.50
  mac: 00:11:22:33:44:55
  width: 2048
  height: 2048
  pixel_format: Mono8

======================================================================
Starting live video stream...
Controls:
  - Press 'q' or ESC to exit
  - Press 's' to save current frame
======================================================================

[Окно OpenCV с видео...]
```

## Структура проекта

```
jai_camera_discovery/
├── main.py                                      # Главный файл приложения
├── connection/
│   ├── ebus_client.py                          # Клиент eBUS SDK
│   └── __init__.py
├── discovery/
│   ├── ebus_discovery.py                       # Обнаружение через eBUS
│   └── __init__.py
├── utils/
│   ├── logger.py                               # Логирование
│   ├── network_utils.py                        # Сетевые утилиты
│   └── __init__.py
├── requirements.txt                            # Python зависимости
├── ebus_python-6.5.4-...-py311-none-win_amd64.whl  # eBUS wheel
└── README.md                                   # Этот файл
```

## Troubleshooting

### Ошибка: "eBUS SDK not available"

**Причина:** eBUS Python API не установлен.

**Решение:**
```bash
py -3.11 -m pip install ebus_python-6.5.4-7277_jai-py311-none-win_amd64.whl
```

Проверьте версию Python:
```bash
py -3.11 --version
```

### Ошибка: "numpy.core.multiarray failed to import"

**Причина:** Установлен NumPy 2.x, а eBUS SDK требует 1.x.

**Решение:**
```bash
py -3.11 -m pip install "numpy<2"
```

Проверьте версию:
```bash
py -3.11 -c "import numpy; print(numpy.__version__)"
```
Должно быть: `1.26.4` или другая версия 1.x.

### Камера не найдена

**Возможные причины:**
1. Камера не подключена или не включена
2. Камера и ПК в разных подсетях
3. Firewall блокирует GigE Vision пакеты
4. Сетевой кабель повреждён

**Решение:**

**Шаг 1:** Проверьте физическое подключение
- Убедитесь, что сетевой кабель подключён к камере и ПК
- Проверьте индикаторы на сетевом порту (должны гореть/мигать)

**Шаг 2:** Проверьте сетевые настройки

Узнайте IP камеры с помощью eBUS Player:
```
C:\Program Files\JAI\eBUS SDK\Binaries\eBUSPlayerJAI64.exe
```

Или используйте JAI Control Tool.

**Шаг 3:** Проверьте подсеть

Камера и ПК должны быть в одной подсети. Например:
- ПК: `192.168.1.100` / `255.255.255.0`
- Камера: `192.168.1.50` / `255.255.255.0`

Проверьте ping:
```bash
ping 192.168.1.50
```

**Шаг 4:** Временно отключите Firewall

```bash
# Проверьте, работает ли обнаружение с выключенным firewall
# Если работает - добавьте исключение для порта 3956 UDP
```

### Чёрное изображение или артефакты

**Возможные причины:**
1. Недостаточное освещение
2. Объектив закрыт крышкой
3. Неправильные настройки экспозиции
4. Неправильный pixel format

**Решение:**
- Проверьте освещение сцены
- Снимите защитную крышку с объектива
- Настройте экспозицию через eBUS Player или JAI Control Tool
- Убедитесь, что pixel format поддерживается (Mono8, Mono12, BayerRG8 и т.д.)

### Низкий FPS или лаги

**Возможные причины:**
1. Используется Fast Ethernet (100 Мбит/с) вместо Gigabit
2. Высокая нагрузка на сеть
3. Слабая производительность ПК

**Решение:**
- Используйте Gigabit Ethernet адаптер и кабель
- Закройте другие программы, использующие сеть
- Уменьшите разрешение или FPS камеры

### OpenCV окно не открывается

**Причина:** Проблемы с OpenCV или дисплеем.

**Решение:**
```bash
# Переустановите OpenCV
py -3.11 -m pip uninstall opencv-python
py -3.11 -m pip install "opencv-python<4.10"

# Проверьте
py -3.11 -c "import cv2; print(cv2.__version__)"
```

## Технические детали

### eBUS SDK

Приложение использует официальный eBUS SDK от Pleora/JAI:
- **Версия SDK:** 6.5.4
- **Python API:** ebus_python
- **Протокол:** GigE Vision (GVCP + GVSP)
- **Совместимость:** JAI RM-4200GE и другие GigE Vision камеры

### Зависимости

| Пакет | Версия | Описание |
|-------|--------|----------|
| Python | 3.11 | Интерпретатор |
| NumPy | 1.26.4 | Обработка массивов (ОБЯЗАТЕЛЬНО <2.0) |
| OpenCV | 4.9.0 | Отображение видео |
| eBUS Python | 6.5.4 | Официальный SDK для JAI камер |

### Технические характеристики JAI RM-4200GE

- **Разрешение:** 2048 x 2048 пикселей (4.2 МП)
- **Интерфейс:** GigE Vision (Gigabit Ethernet)
- **Сенсор:** CMOS
- **Частота кадров:** До 40 fps
- **Pixel formats:** Mono8, Mono10, Mono12, BayerRG8, BayerRG10, BayerRG12
- **Управление:** GVCP (GigE Vision Control Protocol)
- **Передача данных:** GVSP (GigE Vision Streaming Protocol)

## Дополнительная информация

### Официальные ссылки

- **JAI Website:** https://www.jai.com
- **eBUS SDK Downloads:** https://www.jai.com/support-software/jai-software
- **eBUS SDK Documentation:** https://supportcenter.pleora.com/s/article/eBUS-SDK-Python-API-QSG-UG
- **JAI Support:** https://support.jai.com
- **Pleora Technologies:** https://www.pleora.com/machine-vision-connectivity/ebus-sdk/

### Альтернативные инструменты JAI

Для диагностики и настройки камеры:
- **eBUS Player:** `C:\Program Files\JAI\eBUS SDK\Binaries\eBUSPlayerJAI64.exe`
- **JAI Control Tool:** Скачать с https://www.jai.com

## Лицензия

Этот проект использует eBUS SDK от Pleora Technologies Inc.
См. лицензионное соглашение eBUS SDK для подробностей.

## История изменений

### Версия 2.0.0 (Текущая) - eBUS SDK Edition
- Упрощена структура проекта
- Удалены все методы обнаружения кроме eBUS SDK
- Автоматическое обнаружение и подключение
- Автоматический запуск видео стрима
- Простой интерфейс без меню
- Использование официального eBUS SDK

### Версия 1.0.0 (Устаревшая)
- Множественные методы обнаружения (GVCP, Harvester, ARP и т.д.)
- Меню с опциями
- Поддержка Harvester и GenTL Producer
