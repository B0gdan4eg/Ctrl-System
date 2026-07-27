# Архитектура и Pipeline — D:\Ctrl-System

> Обновлено: 2026-05-14 (после аудита и рефакторинга)

---

## Слои приложения

```
┌─────────────────────────────────────────────────────────────┐
│                       main.py (~50 строк)                    │
│           COM STA init → QApplication → DefectDetectionApp  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               UI LAYER  (ui/)  ~1 400 строк                 │
│                                                             │
│  DefectDetectionApp  (main_window.py, ~370 строк)           │
│  ┌────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │ControlPanel│  │ ImageViewer │  │    StatsPanel      │   │
│  │  ~135 стр  │  │  ~104 стр   │  │    ~133 стр        │   │
│  └────────────┘  └─────────────┘  └────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CameraWidget  (~505 строк)                         │    │
│  │  preview │ capture │ stats │ fullscreen              │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
    Qt signals/slots     │  progress / zone_processed / error
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            WORKERS LAYER  (workers/)  ~600 строк            │
│                                                             │
│  ┌───────────────────────┐   ┌──────────────────────────┐  │
│  │  ProcessingWorker     │   │  CameraWorker            │  │
│  │    ~325 строк         │   │  RealtimeCameraWorker    │  │
│  │  QThread              │   │  ContinuousCameraWorker  │  │
│  │  Полный pipeline      │   │    ~275 строк            │  │
│  └───────────┬───────────┘   └──────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│               CORE LAYER  (core/)  ~3 180 строк             │
│                                                             │
│  zone_detection.py  ~375 стр  — find_top_zones, CLAHE       │
│  zone_segmentation.py ~137 стр — U-Net wrapper, lazy load   │
│  image_processing.py  ~330 стр — normalize, preprocess      │
│  defect_detection.py  ~500 стр — predict, refine, GOST      │
│                                                             │
│  cameras/ ~2 240 строк                                      │
│  camera.py(~470) | gige_camera.py(~460)                     │
│  ebus_camera.py(~1100, Pleora) | pylon_camera.py(~207)      │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│               MODELS LAYER  (models/)  ~100 строк           │
│                                                             │
│  UNet: DoubleConv → Encoder(×4) → Bottleneck → Decoder(×4) │
│        → final_conv → sigmoid                               │
│                                                             │
│  best_model.pth   119 MB — детекция дефектов (EMA ep48)    │
│  zone_unet.pth     30 MB — сегментация зон                  │
└─────────────────────────────────────────────────────────────┘

CONFIG: config/settings.py (~60 стр)
UTILS:  utils/logger.py   (~45 стр)
```

---

## Pipeline обработки изображения

```
Пользователь загружает PNG/JPG/TIFF
           │
           ▼  (UI thread)
    [main_window.py]
    Валидация расширения (.png/.jpg/.jpeg/.tiff/.bmp)
    Остановка старого ProcessingWorker (если запущен)
           │
           ▼  (spawn QThread)
    [ProcessingWorker]
           │
    ┌──────▼──────────────────────────────────────────┐
    │  Фаза A: Загрузка и нормализация                │
    │  cv2.imread → convert_to_grayscale               │
    │  apply_histogram_normalization (std_range=2,3)  │
    └──────┬──────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │  Фаза B: Поиск зон                              │
    │  USE_AI_ZONE_DETECTION=True?                    │
    │  ├─YES → ZoneSegmenter.predict_contours() U-Net │
    │  └─NO  → find_top_zones_improved()              │
    │     CLAHE × 30 итераций → _CLAHE_CACHE          │
    │     percentile normalize → contours → top_n     │
    └──────┬──────────────────────────────────────────┘
           │  N зон
    ┌──────▼──────────────────────────────────────────┐
    │  Фаза C: Обработка каждой зоны (цикл)           │
    │  extract_zone_roi(img, contour)                 │
    │  bilateralFilter (1 раз на ROI)                 │
    │  apply_histogram_normalization(std=2.0 и 3.0)   │
    │     last_roi_c кэшируется (без повторного вызова)│
    │  letterbox_resize → prepare_for_display (RGB)   │
    │  ─────────────────────────────────────────────  │
    │  predict_defects_batch_probs()                  │
    │     torch.no_grad() + model.eval() (if training)│
    │     chunk_size=8, TTA (flip H/V)                │
    │     stitch_tile_probs() с косинусным окном      │
    │  ─────────────────────────────────────────────  │
    │  refine_defect_mask()                           │
    │     guided filter × 2 прохода                  │
    │     _intensity_shrink → _SE_CACHE               │
    │     classify_defects_gost() ГОСТ 11141-84       │
    └──────┬──────────────────────────────────────────┘
           │  signal: zone_processed (QueuedConnection)
    ┌──────▼──────────────────────────────────────────┐
    │  Фаза D: Визуализация                           │
    │  draw_zones() — рамки зон                       │
    │  apply_defects_overlay() — красный overlay       │
    │  signal: finished(result_image, stats)          │
    └──────┬──────────────────────────────────────────┘
           │  (UI thread via QueuedConnection)
    ┌──────▼──────────────────────────────────────────┐
    │  [main_window.py]                               │
    │  ImageViewer.set_image(result)                  │
    │  StatsPanel.update_statistics(gost_data)        │
    └─────────────────────────────────────────────────┘
```

---

## Qt Signals/Slots

```
ControlPanel
  ├─→ load_image_clicked      → MainWindow.load_image()
  ├─→ start_processing_clicked→ MainWindow.start_processing()
  └─→ save_result_clicked     → MainWindow.save_result()

ProcessingWorker (QThread)
  ├─→ progress       → StatsPanel.log()
  ├─→ zone_processed → MainWindow.on_zone_processed()
  ├─→ finished       → MainWindow.on_processing_finished()
  └─→ error          → MainWindow.on_processing_error()

CameraCapture (QThread)
  └─→ frame_captured → RealtimeCameraWorker.process_frame()
                       [QueuedConnection — межпоточный]
```

---

## Статистика строк кода (после аудита 2026-05-14)

| Слой | Строк |
|------|-------|
| main.py + config/ + utils/ | ~155 |
| models/ | ~100 |
| core/ (logic: zone, image, defect, segmentation) | ~1 342 |
| core/cameras/ | ~2 237 |
| workers/ | ~600 |
| ui/ (main_window + 4 widgets + styles) | ~1 347 |
| **Production итого** | **~5 780** |
| scripts/ (ML pipeline, tools) | ~4 990 |
| addition/ (experiments) | ~312 |
| **Весь проект итого** | **~11 082** |
| archive/ (мёртвый код, перемещён) | ~345 |

---

## Камеры (поддерживаемые типы)

| Тип | Файл | SDK |
|-----|------|-----|
| USB | `core/cameras/camera.py` | OpenCV |
| GigE (универсальный) | `core/cameras/gige_camera.py` | Harvester/GenICam |
| JAI GigE | `core/cameras/ebus_camera.py` | Pleora eBUS SDK |
| Basler | `core/cameras/pylon_camera.py` | Basler Pylon SDK |
