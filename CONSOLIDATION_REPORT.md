# Отчет о консолидации Ctrl-System — 29.09.2026

## Что сделано
Собрано в один проект `D:\Ctrl-System`, удалены исходники:
- `D:\train` (24.5 ГБ) — удалена целиком
- `D:\AI` (7.5 ГБ) — удалена целиком
- `D:\Ctrl-System v2` (1.6 ГБ) — удалена целиком
- `D:\Ctrl System old` (0.9 ГБ) — удалена целиком
- `D:\jai_camera_discovery` (0.18 ГБ) — удалена целиком (файл `nul` удален через `\\?\`)
- Внутри Ctrl-System удалены: `venv38/`, `dist/`, `dist win/`, `build/`, `archive/`, все `__pycache__/`, старые `logs/defect_detection_*.log` (оставлен последний 20260901), `0`, `dict`, `np.ndarray`, `best_model_backup_ep9_iou5308.pth`, `best_model_old_512.pth.backup`, `Ctrl-System.spec`

Диск D: было ~60.5 ГБ занято → стало 20.6 ГБ. Освобождено ~40 ГБ.

## Что перенесено (важное)
- `training/` — TRAIN_1024_red.py, TRAIN_1024.py, TRAIN.py, train_infer_model.py, prep_train_red.py, prep_train.py, prep_base.py, data_preparation.py, annotate_defects*.py, convert_checkpoint.py, labels_*.json, zone_train.py, zone_model_arch.py, mask_from_json.py, masks_make.py, zone_training_log.json
- `models/collected_best/` — best_model_red_1024_ema.pth, best_model_red_1024.pth, best_model_1024.pth, best_zone_unet.pth (+ штатные best_model.pth в корне и models/zone_unet.pth)
- `scripts/surface_v2/` — calibrate_multispectral.py, run_pipeline.py, surface_of_unfitness.py, surface_maps.py, surface_3d_interactive.py, whole_filter_3d.py, surface_ui.py
- `workers/energy_worker.py`, `workers/surface_worker.py`
- `ui/widgets/surface_3d_widget.py`
- `config/settings_v2_surface.py`
- `docs/v2_docs/` — architecture_v2.md, brightness_verdict_design.md, tasks_progress.md, roadmap_stage_stitching.md, audit_report_2026_05_14.md + docs/architecture_old_v1.md
- `core/cameras/_jai_ebus_client.py` (из jai connection/ebus_client.py)
- `utils/network_utils_jai.py`
- `hardware/ebus_sdk/` — jai_main_reference.py, jai_requirements.txt, README.md, PROJECT_STATUS.md, ebus_python-6.5.4 whl, eBUS Quick Start PDF

## Структура Ctrl-System сейчас
- main.py, requirements.txt, readme.md, icon.ico, .gitignore, best_model.pth
- config/, core/, workers/, ui/, utils/, models/, scripts/, training/, hardware/, docs/, logs/, results/, uploads/, test/, addition/, WTS 05.08.2026/
- .git/, .claude/, .vscode/ — оставлены

## Внимание
- Датасеты prepared_*/zone_dataset_full удалены вместе с train/AI. Остались только best-модели и labels_*.json.
- `.git/objects` тяжелый (история с бинарниками) — оставлен, при желании `git gc --aggressive` или переинит.
- `WTS 05.08.2026/`, `test/`, `addition/zone_detection_dev.py` — оставлены, проверить вручную.
- eBUS нативный код (`_jai_ebus_client.py`, `network_utils_jai.py`) скопирован как референс, в `core/cameras/ebus_camera.py` пока старый PvDotNet-вариант — объединить вручную.
- Запуск: `python main.py` (нужно свежее venv, requirements.txt).
