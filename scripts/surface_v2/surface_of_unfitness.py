"""
v2 прототип: 3D-карта яркости + "поверхность негодности" (design §v2.1, v2.3).

Классический подход (без нейросети, GPU не нужен — см. design v2.4):
  фон B(x,y)  — grayscale closing (заполняет тёмные дефекты) + сглаживание
  шум  sigma  — модель sigma^2(B) = read^2 + B/gain (PTC: наклон дисперсия-vs-среднее)
  поверхность = B - k*sigma          ← адаптивный порог, следует за фоном
  дефект      = там, где рельеф img протыкает поверхность снизу
  p%          = (B - img) / B * 100  ← глубина протыкания (отклонение яркости)
  S(x,y)      = плотность глубины-в-сигмах  → шип / бугор / рябь (кластеры)

Работает на сырых 16-бит снимках (uint16, 12-бит сенсор). Кириллические пути читаются
через np.fromfile+imdecode (cv2.imread их не открывает на Windows).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


# --------------------------------------------------------------------------- IO
def imread_u16(path: str) -> np.ndarray:
    """Cyrillic-safe чтение в одноканальный массив (uint16, как снято)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot decode image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def imwrite_u(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


def find_dark_frame(image_path: str) -> str | None:
    """Ищет тёмновой кадр (Test_1_0.png в соседней папке 'Темновой') рядом со снимком."""
    specimen = os.path.dirname(os.path.dirname(image_path))  # .../№N/<band>/file -> .../№N
    for root, _dirs, files in os.walk(specimen):
        if "темнов" in os.path.basename(root).lower():
            for f in files:
                if f.lower().endswith(".png"):
                    return os.path.join(root, f)
    return None


# ----------------------------------------------------------------------- zone
def segment_zone(img16: np.ndarray, erode_margin: int = 9) -> np.ndarray:
    """Маска яркой световой зоны: percentile-нормализация -> Otsu -> крупнейший контур.

    erode_margin отрезает яркостный спад у границы зоны, чтобы он не отравил фон/шум.

    Это ЕДИНСТВЕННЫЙ доступный способ найти зону, когда на входе всего ОДИН снимок
    (standalone CLI --image ниже) — в отличие от полного пайплайна (run_pipeline.py /
    calibrate_multispectral.detect_zones), которому доступны все спектральные полосы сразу
    и который поэтому находит все 5 зон разом. На одном снимке границы могут получиться
    другими, чем при прогоне через полный пайплайн той же детали — это ожидаемо, не баг:
    для результатов, сравнимых с GUI/run_pipeline.py, гоняйте деталь целиком, а не один файл.
    """
    lo, hi = np.percentile(img16, [1, 99])
    norm = np.clip((img16.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    _t, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask.astype(bool)
    big = max(cnts, key=cv2.contourArea)
    zone = np.zeros_like(mask)
    cv2.drawContours(zone, [big], -1, 255, thickness=cv2.FILLED)
    if erode_margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_margin, erode_margin))
        zone = cv2.erode(zone, k)
    return zone.astype(bool)


# ----------------------------------------------------------------- background
def estimate_background(img16: np.ndarray, downscale: int = 4, med_size: int = 15,
                        smooth: int = 21, zone: np.ndarray | None = None) -> np.ndarray:
    """Фон B(x,y) = медиана недефектных пикселей зоны (design §7.4).

    Медиана робастна к редким тёмным дефектам (в отличие от closing, который тянет
    к локальному максимуму текстуры и завышает фон). Большое медианное окно на полном
    2048² дорого и cv2.medianBlur для uint16 ограничен ksize<=5, поэтому считаем медиану
    на уменьшенной копии и возвращаем апсемплингом — плато крупномасштабное, деталей не теряем.

    zone: если задана, вне-зонные пиксели заполняются ближайшим значением зоны
    (nearest-fill), иначе тёмный фон вокруг зоны протекает в окно медианы у краёв и
    занижает B → ложное «ярче фона» кольцо по периметру зоны.

    ВНИМАНИЕ: медиана робастна только к МАЛЫМ дефектам — пятно шириной ~медианного
    окна занимает всё окно, B следует за ним, коридор ДИ провисает и ядро дефекта
    сливается с фоном (образец мод в/№6: пятно ~200px в зоне 2). Для полос фильтра
    используйте estimate_background_strip — он к широким пятнам робастен по построению.
    """
    H, W = img16.shape
    src = img16.astype(np.float32)
    if zone is not None and not zone.all():
        try:
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~zone, return_distances=False, return_indices=True)
            src = src[tuple(idx)]
        except Exception:
            fill = float(np.median(src[zone])) if zone.any() else 0.0
            src = np.where(zone, src, fill)
    small = cv2.resize(src, (W // downscale, H // downscale),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    med_size = med_size + 1 if med_size % 2 == 0 else med_size
    try:
        from scipy.ndimage import median_filter
        bs = median_filter(small, size=med_size)
    except Exception:  # fallback без scipy — closing (менее точно)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (med_size, med_size))
        bs = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k)
    bg = cv2.resize(bs, (W, H), interpolation=cv2.INTER_LINEAR)
    if smooth > 1:
        bg = cv2.blur(bg, (smooth, smooth))
    return bg.astype(np.float32)


def _profile_fill_smooth(v: np.ndarray, med: int = 31) -> np.ndarray:
    """1D-профиль: заполнить NaN (столбцы/строки вне зоны) интерполяцией и сгладить медианой."""
    n = v.shape[0]
    idx = np.flatnonzero(~np.isnan(v))
    if idx.size == 0:
        return np.zeros(n, np.float32)
    if idx.size < n:  # края наружу — крайними валидными значениями
        out = np.interp(np.arange(n, dtype=np.float32), idx, v[idx]).astype(np.float32)
    else:
        out = v.copy()
    k = med + 1 if med % 2 == 0 else med
    try:
        from scipy.ndimage import median_filter
        out = median_filter(out, size=k)
    except Exception:  # fallback без scipy
        out = cv2.blur(out.reshape(1, -1), (k, 1)).ravel()
    return out.astype(np.float32)


def _robust_poly_profile(v: np.ndarray, deg: int = 3, n_iter: int = 2,
                         clip: float = 2.5) -> np.ndarray:
    """1D-профиль: робастная полиномиальная аппроксимация с сигма-отсечением.

    Медианное сглаживание окном med убирает провалы короче med/2, но самая тёмная
    полоса широкого пятна длится и >100 строк (№6, зона 2: 115 строк, −131 DN) —
    а окно >230 строк уже искажает полезный градиент. Полином + отсечение решает
    обе задачи сразу: плавный градиент полос/освещения ловится степенью deg, а
    локальные провалы и пики ЛЮБОЙ длины отбрасываются как выбросы итерациями.
    """
    n = v.shape[0]
    idx = np.flatnonzero(~np.isnan(v))
    if idx.size == 0:
        return np.zeros(n, np.float32)
    if idx.size < max(2 * deg + 2, 12):     # слишком короткий профиль — без аппроксимации
        return _profile_fill_smooth(v, med=31)
    y = idx.astype(np.float64)
    q = v[idx].astype(np.float64)
    sel = np.ones(idx.size, bool)
    c = np.polyfit(y, q, deg)
    for _ in range(n_iter + 1):
        c = np.polyfit(y[sel], q[sel], deg)
        r = q - np.polyval(c, y)
        s = float(np.std(r[sel]))
        if s < 1e-3:
            break
        keep = np.abs(r) < clip * s
        if keep.sum() < max(2 * deg + 2, 12) or np.array_equal(keep, sel):
            break
        sel = keep
    out = np.polyval(c, np.arange(n, dtype=np.float64))
    return out.astype(np.float32)


def estimate_background_strip(img16: np.ndarray, zone: np.ndarray,
                              p_y: float = 75.0, p_x: float = 95.0,
                              med: int = 31) -> np.ndarray:
    """Фон полосы как сепарабельный профиль P(x) + Q(y) — робастный к ШИРОКИМ дефектам.

    estimate_background (медиана) проваливается, когда пятно шириной ~медианного окна:
    дефект занимает всё окно, B следует за ним, коридор ДИ провисает над ямой, а ядро
    дефекта сливается с фоном и не детектируется (образец мод в/№6: размытое пятно
    ~200px в зоне 2 не было найдено вовсе). Физика полосы фильтра: уровень фона плавен
    ВДОЛЬ полосы, локальные 2D-провалы — дефекты, а не фон. Отсюда оценка:
      P(x) — перцентиль p_y по столбцу в пределах зоны. Пятно занимает лишь часть
             высоты полосы, перцентиль выше этой доли видит чистый уровень (смещение
             вверх от шума ~0.7·σ ≈ 1-2% — много меньше p_min=20%);
      Q(y) — перцентиль p_x по строке от остатка (img − P): возвращает плавный градиент
             вдоль полосы (зона 4 №6 градиент ~200→450 DN) и не проваливается в
             широкие тёмные пятна: они могут занимать большинство ШИРИНЫ строки
             (пятно №6 — 87% ширины зоны при 15% её высоты), поэтому p_x должен
             быть высоким (смещение вверх ~1-2% несущественно). Остаточные провалы
             там, где пятно перекрывает > (100−p_x)% ширины, добивает робастная
             полиномиальная аппроксимация Q с сигма-отсечением (_robust_poly_profile):
             медианное окно не спасает — тёмная полоса пятна тянется >100 строк.
    Итог B(x,y) = P(x) + Q(y); вне зоны значения заполняются ближайшими (nearest-fill).

    Ограничение: дефект, занимающий > (100−p_y)% высоты И > (100−p_x)% ширины зоны,
    фоном всё же станет — но такая «засветка» уже смена уровня зоны, а не дефект.
    """
    H, W = img16.shape
    f = img16.astype(np.float32)
    rows_any = np.flatnonzero(zone.any(axis=1))
    cols_any = np.flatnonzero(zone.any(axis=0))
    if rows_any.size == 0 or cols_any.size == 0:
        return estimate_background(img16, zone=zone)
    y0, y1 = int(rows_any[0]), int(rows_any[-1]) + 1
    x0, x1 = int(cols_any[0]), int(cols_any[-1]) + 1
    sub = f[y0:y1, x0:x1]
    zsub = zone[y0:y1, x0:x1]

    P = np.full(sub.shape[1], np.nan, np.float32)
    for j in range(sub.shape[1]):           # столбцы зоны
        col = zsub[:, j]
        if col.sum() >= 30:
            P[j] = np.percentile(sub[col, j], p_y)
    P = _profile_fill_smooth(P, med)

    Q = np.full(sub.shape[0], np.nan, np.float32)
    R = sub - P[None, :]
    for i in range(sub.shape[0]):           # строки зоны
        row = zsub[i, :]
        if row.sum() >= 30:
            Q[i] = np.percentile(R[i, row], p_x)
    Q = _robust_poly_profile(Q)

    Bsub = P[None, :] + Q[:, None]
    B = np.empty((H, W), np.float32)
    B[y0:y1, x0:x1] = Bsub
    if not zone.all():                       # вне зоны — ближайшие значения зоны
        try:
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~zone, return_distances=False, return_indices=True)
            B = B[tuple(idx)]
        except Exception:
            fill = float(np.median(Bsub[zsub])) if zsub.any() else 0.0
            B = np.where(zone, B, fill)
    return B


# --------------------------------------------------------------- noise model
@dataclass
class NoiseModel:
    read2: float   # read^2, intercept (DN^2)
    inv_gain: float  # 1/gain, slope (DN per DN)  -> sigma^2 = read2 + inv_gain * B

    def sigma(self, B: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.read2 + self.inv_gain * B, 1e-6))


def _tile_stats(img16: np.ndarray, zone: np.ndarray, B: np.ndarray,
                tile: int = 32) -> tuple[list[float], list[float]]:
    """Дисперсия остатка (img-B) по тайлам, только внутри зоны. Дефекты только ДОБАВЛЯЮТ
    дисперсию — поэтому шумовой пол берётся ниже (25-й перцентиль по бинам), см. _fit_ptc."""
    f = img16.astype(np.float32)
    res = f - B
    H, W = f.shape
    bvals, rvar = [], []
    for y in range(0, H - tile, tile):
        for x in range(0, W - tile, tile):
            if not zone[y:y + tile, x:x + tile].all():
                continue
            bvals.append(float(B[y:y + tile, x:x + tile].mean()))
            rvar.append(float(res[y:y + tile, x:x + tile].var()))
    return bvals, rvar


def _read2_from_dark(dark_path: str) -> float:
    """read^2 из тёмнового кадра: ЛОКАЛЬНЫЙ высокочастотный шум (МAD после вычитания
    сглаженного кадра), а не дисперсия по всему кадру целиком. Дисперсия всего кадра
    смешивает истинный temporal read-noise с фикс.-паттерн шумом/градиентом освещения/
    горячими пикселями и завышает read — MAD после detrend устойчив к этим выбросам."""
    dark = imread_u16(dark_path).astype(np.float32)
    smooth = cv2.GaussianBlur(dark, (0, 0), 5)
    resid = dark - smooth
    mad = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
    return mad * mad


def _fit_ptc(bvals: list[float], rvar: list[float], mad: float, b_med: float,
            read2: float | None) -> NoiseModel:
    """Общий МНК-фит sigma^2(B) = read^2 + B/gain по (bvals, rvar), с MAD-фолбэком при
    вырожденных/скудных данных. Вынесено отдельно, чтобы одинаково работать и на одном
    кадре (estimate_noise_model), и на пуле тайлов сразу со всех зон (estimate_noise_model_pooled)."""
    if len(bvals) >= 8:
        bv_arr = np.asarray(bvals); rv_arr = np.asarray(rvar)
        nb = 12
        edges = np.linspace(bv_arr.min(), bv_arr.max(), nb + 1)
        bm, bvv = [], []
        for i in range(nb):
            sel = (bv_arr >= edges[i]) & (bv_arr < edges[i + 1])
            if sel.sum() >= 3:
                bm.append(bv_arr[sel].mean())
                bvv.append(np.percentile(rv_arr[sel], 25))  # шумовой пол
        if len(bm) >= 2:
            slope, intercept = np.polyfit(np.asarray(bm), np.asarray(bvv), 1)
        else:
            slope, intercept = mad * mad / max(b_med, 1.0), mad * mad
    else:
        slope, intercept = mad * mad / max(b_med, 1.0), mad * mad

    if read2 is None:
        read2 = max(intercept, 1.0)
    # наклон вырожден (<=0) — восстанавливаем из MAD так, чтобы sigma(b_med)=mad
    if slope <= 1e-4:
        slope = max((mad * mad - read2) / max(b_med, 1.0), 1e-4)
    return NoiseModel(read2=max(read2, 1.0), inv_gain=float(slope))


def estimate_noise_model(img16: np.ndarray, zone: np.ndarray, B: np.ndarray,
                         dark_path: str | None = None, tile: int = 32) -> NoiseModel:
    """Photon-transfer-curve: sigma^2(B) = read^2 + B/gain, по ОДНОМУ кадру/зоне.

    ВНИМАНИЕ: у одной зоны диапазон яркости B узкий (полоса уже выбрана "ближайшей к 2000"),
    поэтому подгонка наклона (dispersion vs mean) по единственной зоне статистически шаткая
    (мало разброса B). Design §4 предполагал калибровку read/gain ОДИН РАЗ по широкому
    диапазону — используйте estimate_noise_model_pooled() по всем зонам детали разом, если
    они доступны (run_pipeline.py, surface_3d_interactive.py). Эта функция — фолбэк для
    случая, когда доступен только один кадр/одна зона (standalone --image CLI ниже).
    """
    bvals, rvar = _tile_stats(img16, zone, B, tile)
    rz = (img16.astype(np.float32) - B)[zone]
    mad = float(np.median(np.abs(rz - np.median(rz))) * 1.4826)
    b_med = float(np.median(B[zone]))
    read2 = _read2_from_dark(dark_path) if (dark_path and os.path.exists(dark_path)) else None
    return _fit_ptc(bvals, rvar, mad, b_med, read2)


def estimate_noise_model_pooled(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                                dark_path: str | None = None, tile: int = 32) -> NoiseModel:
    """Тот же PTC-фит, но тайлы собираются сразу со ВСЕХ зон детали:
    samples = [(img16, zone, B), ...] — одна запись на зону (её лучшая калиброванная
    полоса + её фон B). У разных зон разная яркость фона, поэтому пул перекрывает широкий
    диапазон B за один проход — наклон sigma^2=read^2+B/gain оценивается устойчивее, чем
    по одной зоне (design §4: read/gain калибруются один раз на деталь, не на зону)."""
    bvals, rvar, resid_all, b_all = [], [], [], []
    for img16, zone, B in samples:
        bv, rv = _tile_stats(img16, zone, B, tile)
        bvals.extend(bv); rvar.extend(rv)
        res = (img16.astype(np.float32) - B)[zone]
        resid_all.append(res)
        b_all.append(B[zone])
    if not resid_all:
        return NoiseModel(read2=1.0, inv_gain=1e-4)
    rz = np.concatenate(resid_all)
    mad = float(np.median(np.abs(rz - np.median(rz))) * 1.4826)
    b_med = float(np.median(np.concatenate(b_all)))
    read2 = _read2_from_dark(dark_path) if (dark_path and os.path.exists(dark_path)) else None
    return _fit_ptc(bvals, rvar, mad, b_med, read2)


# ------------------------------------------------------------------ detection
@dataclass
class Result:
    img: np.ndarray
    zone: np.ndarray
    B: np.ndarray
    sigma: np.ndarray
    surface: np.ndarray     # эффективная поверхность негодности = min(B-kσ, ДИ 0.8·B)
    di_dark: np.ndarray     # ДИ-граница тёмного p≥p_min: B·(1-p_min/100), привязана к фону зоны
    di_white: np.ndarray    # ДИ-граница белого p≥p_min: B·(1+p_min/100)
    dark_mask: np.ndarray   # тёмные дефекты (рельеф протыкает поверхность снизу)
    white_mask: np.ndarray  # белые дефекты (ярче фона на k*sigma)
    p_map: np.ndarray       # отклонение яркости % (темнее фона), только в дефектах
    wp_map: np.ndarray      # отклонение яркости % (ярче фона), только в белых дефектах
    depth_sigma: np.ndarray # глубина в сигмах (для слоя плотности)
    density: np.ndarray     # S(x,y): шип/бугор/рябь
    noise: NoiseModel


def _clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Убирает компоненты меньше min_area px (одиночный шот-шум; min дефект ≥3px, v2.5)."""
    if min_area <= 1:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out.astype(bool)


def process(img16: np.ndarray, k: float = 4.0, bg_kernel: int = 15,
            density_sigma: float = 15.0, dark_path: str | None = None,
            zone: np.ndarray | None = None, p_min: float = 20.0,
            min_area: int = 3, noise: "NoiseModel | None" = None,
            strip_bg: bool = False) -> Result:
    """noise: готовая шумовая модель (см. estimate_noise_model_pooled) — если задана,
    read/gain НЕ переоцениваются заново для этой зоны (design §4: калибруется один раз
    на деталь). Если None — оцениваем локально по этому кадру/зоне (фолбэк, менее
    устойчиво на узком диапазоне B одной зоны, см. estimate_noise_model).

    strip_bg: фон полосы estimate_background_strip (P(x)+Q(y), робастный к широким
    пятнам) вместо медианного estimate_background — для вертикальных полос фильтра."""
    if zone is None:
        zone = segment_zone(img16)
    if strip_bg:
        B = estimate_background_strip(img16, zone)
    else:
        B = estimate_background(img16, med_size=bg_kernel, zone=zone)
    if noise is None:
        noise = estimate_noise_model(img16, zone, B, dark_path=dark_path)
    sigma = noise.sigma(B)

    # ДИ-поверхности негодности (критерий протокол.docx): порог в DN привязан к фону
    # КОНКРЕТНОЙ зоны, т.к. p% относителен. У яркой зоны 0.8·B высоко, у тусклой — низко.
    di_dark = B * (1.0 - p_min / 100.0)        # тёмный/значимый  p≥p_min  (по умолч. 0.8·B)
    di_white = B * (1.0 + p_min / 100.0)        # белый           p≥p_min  (1.2·B)

    # эффективная поверхность = min(шумовой пол B−kσ, ДИ-порог 0.8·B):
    # дефект там, где рельеф ниже ОБЕИХ — k·σ глушит шум в тусклых зонах, ДИ задаёт
    # критерий годности в ярких. f<surface ⇔ (f<B−kσ И p≥p_min) — гейт и критерий слиты.
    noise_floor = B - k * sigma
    surface = np.minimum(noise_floor, di_dark)
    surface_white = np.maximum(B + k * sigma, di_white)

    f = img16.astype(np.float32)
    Bsafe = np.maximum(B, 1e-3)
    p_dark = np.where(zone, (B - f) / Bsafe * 100.0, 0.0)   # >0 темнее фона
    p_white = np.where(zone, (f - B) / Bsafe * 100.0, 0.0)  # >0 ярче фона

    dark_mask = _clean_mask((f < surface) & zone, min_area)
    white_mask = _clean_mask((f > surface_white) & zone, min_area)

    p_map = np.where(dark_mask, p_dark, 0.0)
    wp_map = np.where(white_mask, p_white, 0.0)

    depth_sigma = np.maximum(np.where(zone, (B - f) / np.maximum(sigma, 1e-6), 0.0), 0.0)
    # S(x,y): плотность глубины-в-сигмах, ТОЛЬКО от реальных дефектов (dark_mask). Без этой
    # маски шумовые флуктуации по ~половине зоны (везде где f<B) дают ложный пьедестал и
    # чистая зона "вспучивается" даже без единого дефекта (баг был найден и исправлен
    # 2026-06-11, но только в вызывающем коде surface_3d_interactive.build_maps — здесь,
    # в источнике, тот фикс не применялся, из-за чего summarize()/save_panels() ниже его
    # не видели).
    ksz = int(density_sigma * 6) | 1
    density = cv2.GaussianBlur(np.where(dark_mask, depth_sigma, 0.0).astype(np.float32),
                               (ksz, ksz), density_sigma)

    return Result(img16, zone, B, sigma, surface, di_dark, di_white,
                  dark_mask, white_mask, p_map, wp_map, depth_sigma, density, noise)


# --------------------------------------------------------------------- verdict
def summarize(r: Result) -> dict:
    """Предварительные бюджеты пикселей (design §1). ПК/МК-классификация зон — TODO."""
    p = r.p_map
    return {
        "zone_px": int(r.zone.sum()),
        "dark_defect_px": int(r.dark_mask.sum()),
        "px_p_gt_20": int(((p >= 20) & (p < 50)).sum()),
        "px_p_gt_50": int((p >= 50).sum()),
        "white_defect_px": int(r.white_mask.sum()),
        "cluster_peak_S": round(float(r.density.max()), 2),
        "read": round(float(np.sqrt(r.noise.read2)), 2),
        "inv_gain(slope)": round(float(r.noise.inv_gain), 4),
    }


# ----------------------------------------------------------------- visualize
def _to_u8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def save_panels(r: Result, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = _to_u8(r.img.astype(np.float32))
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    rgb[r.dark_mask] = (0, 0, 255)     # тёмные дефекты — красным
    rgb[r.white_mask] = (0, 255, 255)  # белые дефекты — жёлтым
    imwrite_u(os.path.join(out_dir, f"{name}_overlay.png"), rgb)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax[0, 0].imshow(base, cmap="gray"); ax[0, 0].set_title("Исходный (зона)")
    ax[0, 1].imshow(r.B, cmap="viridis"); ax[0, 1].set_title("Фон B(x,y)")
    ax[0, 2].imshow(r.sigma, cmap="magma"); ax[0, 2].set_title("Шум sigma(x,y)")
    pm = np.where(r.p_map > 0, r.p_map, np.nan)
    im3 = ax[1, 0].imshow(pm, cmap="hot", vmin=0, vmax=60)
    ax[1, 0].set_title("Отклонение p%"); fig.colorbar(im3, ax=ax[1, 0], fraction=0.046)
    im4 = ax[1, 1].imshow(r.density, cmap="inferno")
    ax[1, 1].set_title("Плотность S(x,y): шип/бугор/рябь")
    fig.colorbar(im4, ax=ax[1, 1], fraction=0.046)
    ax[1, 2].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax[1, 2].set_title("Дефекты (красн=тёмн, жёлт=бел)")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_panels.png"), dpi=110)
    plt.close(fig)


def save_surface3d(r: Result, out_dir: str, name: str, step: int | None = None) -> None:
    """3D: рельеф яркости (img) и поверхность негодности (B-k*sigma)."""
    H, W = r.img.shape
    if step is None:
        step = max(1, max(H, W) // 160)  # ~160x160 для скорости отрисовки
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    Z_img = r.img[::step, ::step].astype(np.float32)
    Z_surf = r.surface[::step, ::step]
    zmask = r.zone[::step, ::step]
    Z_img = np.where(zmask, Z_img, np.nan)
    Z_surf = np.where(zmask, Z_surf, np.nan)

    fig = plt.figure(figsize=(13, 9))
    axp = fig.add_subplot(111, projection="3d")
    axp.plot_surface(X, Y, Z_img, cmap=cm.gray, alpha=0.85,
                     linewidth=0, antialiased=False, rstride=1, cstride=1)
    axp.plot_surface(X, Y, Z_surf, color="red", alpha=0.25,
                     linewidth=0, antialiased=False, rstride=1, cstride=1)
    axp.set_title("Рельеф яркости (серый) vs поверхность негодности по ДИ min(B-k·σ, 0.8·B) (красная)")
    axp.set_xlabel("X"); axp.set_ylabel("Y"); axp.set_zlabel("яркость (DN)")
    axp.view_init(elev=35, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_surface3d.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="v2 поверхность негодности (прототип)")
    ap.add_argument("--image", required=True, help="путь к 16-бит снимку (.png)")
    ap.add_argument("--dark", default=None, help="тёмновой кадр (по умолчанию авто-поиск)")
    ap.add_argument("--k", type=float, default=4.0, help="порог в сигмах (k·σ)")
    ap.add_argument("--bg-kernel", type=int, default=15, help="окно медианы фона (downsampled)")
    ap.add_argument("--density-sigma", type=float, default=15.0, help="окно плотности S")
    ap.add_argument("--out", default=None, help="папка вывода")
    args = ap.parse_args()

    img = imread_u16(args.image)
    dark = args.dark or find_dark_frame(args.image)
    name = os.path.splitext(os.path.basename(args.image))[0]
    out = args.out or os.path.join("results", "v2_surface", name)

    print(f"[i] image {img.shape} {img.dtype} range {img.min()}..{img.max()}")
    print(f"[i] dark frame: {dark if dark else 'нет (read оценим по PTC)'}")

    r = process(img, k=args.k, bg_kernel=args.bg_kernel,
                density_sigma=args.density_sigma, dark_path=dark)
    save_panels(r, out, name)
    save_surface3d(r, out, name)

    print("[verdict] " + "  ".join(f"{kk}={vv}" for kk, vv in summarize(r).items()))
    print(f"[ok] saved to {out}")


if __name__ == "__main__":
    main()
