"""
v2 полный пайплайн на одну деталь:
  калибровка (dark/flat под источник) -> детекция зон -> лучшая полоса на зону
  (без насыщения) -> поверхность негодности по каждой зоне -> вердикт + визуализация.

Запуск:
  python scripts/run_pipeline.py --specimen ".../№N" [--k 4]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from surface_of_unfitness import (  # noqa: E402
    process, imread_u16, imwrite_u, _to_u8, estimate_background_strip,
    estimate_noise_model_pooled,
)
from calibrate_multispectral import (  # noqa: E402
    discover_frames, load_calibrated, detect_zones_multiband, zone_mask, best_band, SAT,
)

ERODE = 25         # эрозия маски зоны для анализа (срезает яркостный спад у границы)
# доля насыщенных пикселей, выше которой зона "не измерима" — это ФИНАЛЬНЫЙ гейт (терпимее,
# чем 2%-й порог выбора полосы в best_band(): здесь бракуем измеримость зоны целиком, только
# если даже её лучшая полоса пересвечена больше чем на 10%).
SAT_ZONE = 0.10

# Бюджеты пикселей по категориям (brightness_verdict_design.md §1, таблица допусков).
# Тип зоны определяем по её "родной"/лучшей полосе: панхром -> ПК, остальные 4 полосы
# (красный/ИК/зелёный/синий) -> МК (§7.1: раскладка зон 1:1 совпадает с 5 полосами).
# "Кластер" и "Значительный" дефект (геометрическая классификация компонент по протокол.docx,
# не по одному лишь p%) пока НЕ реализованы отдельно — сюда не фабрикуем произвольный
# геометрический критерий без реального текста протокола. Бюджет "общее" (150) частично
# страхует от их пропуска, но это не полная замена — см. docs/tasks_progress.md.
ZONE_TYPE_BY_BAND = {"панхром": "ПК"}  # все прочие полосы -> МК (см. .get(..., "МК") ниже)
BUDGETS = {
    "ПК": {"point": 100, "deep": 50, "white": 2, "total": 150},
    "МК": {"point": 100, "deep": 50, "white": 4, "total": 150},
}


def zone_type_of(band: str) -> str:
    return ZONE_TYPE_BY_BAND.get(band, "МК")


def verdict_for(zone_type: str, p20: int, p50: int, white_px: int) -> tuple[str, str]:
    """Вердикт по бюджетам пикселей (не "любой дефектный пиксель = брак"). Возвращает
    (verdict, reason) — reason перечисляет какие именно бюджеты превышены, для прозрачности."""
    b = BUDGETS[zone_type]
    total = p20 + p50
    reasons = []
    if p20 > b["point"]:
        reasons.append(f"точечный {p20}>{b['point']}")
    if p50 > b["deep"]:
        reasons.append(f"глубокий {p50}>{b['deep']}")
    if white_px > b["white"]:
        reasons.append(f"белые {white_px}>{b['white']}")
    if total > b["total"]:
        reasons.append(f"общее {total}>{b['total']}")
    return ("БРАК", "; ".join(reasons)) if reasons else ("годен", "")


def run(spec: str, k: float = 4.0) -> dict:
    frames = discover_frames(spec)
    cal, raw = load_calibrated(frames)
    proj = np.stack(list(cal.values())).max(0)
    zones = detect_zones_multiband(proj, cal, raw)

    H, W = proj.shape
    overlay = cv2.cvtColor(_to_u8(proj), cv2.COLOR_GRAY2BGR)
    pmap_all = np.zeros((H, W), np.float32)
    rows = []
    defects = []  # (zone_idx, band, frame, Result) для зон с дефектами -> 3D

    # --- проход 1: геометрия/полоса/насыщение на зону + фон B -> общий пул для ОДНОЙ
    # шумовой модели на всю деталь (design §4: read/gain калибруются один раз, а не
    # заново на узком диапазоне яркости каждой отдельной зоны).
    zinfo = []
    for i, c in enumerate(zones):
        x, y, w, h = cv2.boundingRect(c)
        zm = zone_mask(proj.shape, c, erode=ERODE)
        sel = best_band(zm, raw)
        band = sel["best"]
        frame = cal[band]
        sat_frac = float((raw[band][zm] >= SAT).mean())
        median = round(sel["bands"][band]["median"], 0)  # сырая медиана (база окна 2000±250)
        zinfo.append(dict(i=i, x=x, zm=zm, sel=sel, band=band, frame=frame,
                          sat_frac=sat_frac, median=median))

    samples = []
    for zi in zinfo:
        if zi["sat_frac"] > SAT_ZONE:
            continue
        zi["B"] = estimate_background_strip(zi["frame"], zone=zi["zm"])
        samples.append((zi["frame"], zi["zm"], zi["B"]))
    noise = estimate_noise_model_pooled(samples, dark_path=frames["dark"]) if samples else None

    # --- проход 2: детекция + бюджетный вердикт. strip_bg=True: фон зоны —
    # сепарабельный профиль полосы (estimate_background_strip), робастный к широким
    # пятнам: медианный фон провисает над пятном шириной ~медианного окна, коридор
    # ДИ «прощает» его ядро и вердикт недосчитывает пиксели (образец мод в/№6).
    for zi in zinfo:
        i, x, zm, sel, band, frame = zi["i"], zi["x"], zi["zm"], zi["sel"], zi["band"], zi["frame"]
        sat_frac, median = zi["sat_frac"], zi["median"]

        # зона насыщена во всех полосах → измерять нельзя (потолок, дефект невидим, §3)
        if sat_frac > SAT_ZONE:
            overlay[zm] = (255, 0, 255)  # пурпур = не измеримо (насыщение)
            rows.append({"zone": i, "x": x, "band": band, "type": zone_type_of(band),
                         "zpx": int(zm.sum()), "median": median, "dark_px": 0, "p20": 0,
                         "p50": 0, "white_px": 0, "sat": round(sat_frac, 3),
                         "verdict": "насыщ", "reason": ""})
            continue

        r = process(frame, k=k, zone=zm, dark_path=frames["dark"], noise=noise, strip_bg=True)
        overlay[r.dark_mask] = (0, 0, 255)    # тёмные дефекты — красным
        overlay[r.white_mask] = (0, 255, 255)  # белые — жёлтым
        pmap_all = np.maximum(pmap_all, r.p_map)

        p20 = int(((r.p_map >= 20) & (r.p_map < 50)).sum())
        p50 = int((r.p_map >= 50).sum())
        white_px = int(r.white_mask.sum())
        ztype = zone_type_of(band)
        verdict, reason = verdict_for(ztype, p20, p50, white_px)
        if not sel.get("in_window", True):
            verdict += "?"  # зона вне окна 2000±250 — измерение менее достоверно
        rows.append({"zone": i, "x": x, "band": band, "type": ztype, "zpx": int(zm.sum()),
                     "median": median, "dark_px": int(r.dark_mask.sum()),
                     "p20": p20, "p50": p50, "white_px": white_px,
                     "sat": round(sat_frac, 3), "verdict": verdict, "reason": reason})
        if r.dark_mask.any() or r.white_mask.any():
            defects.append((i, band, frame, r))
    return {"overlay": overlay, "pmap": pmap_all, "proj": proj,
            "rows": rows, "defects": defects}


def save_defect_3d(frame, r, out: str, name: str, zi: int, band: str, rad: int = 40) -> None:
    """Локальный 3D вокруг дефекта: рельеф яркости (серый) протыкает поверхность B-k·σ (красная)."""
    ys, xs = np.where(r.dark_mask | r.white_mask)
    if len(xs) == 0:
        return
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, y1 = max(0, cy - rad), min(frame.shape[0], cy + rad)
    x0, x1 = max(0, cx - rad), min(frame.shape[1], cx + rad)
    X, Y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, frame[y0:y1, x0:x1], cmap=cm.gray, alpha=0.9,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.plot_surface(X, Y, r.surface[y0:y1, x0:x1], color="red", alpha=0.30, linewidth=0)
    ax.set_title(f"{name} z{zi} ({band}): рельеф протыкает поверхность негодности B-k·σ (красная)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("яркость DN")
    ax.view_init(elev=22, azim=-72)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_z{zi}_3d.png"), dpi=120)
    plt.close(fig)


def save(res: dict, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    imwrite_u(os.path.join(out, f"{name}_overlay.png"), res["overlay"])
    for zi, band, frame, r in res.get("defects", []):
        save_defect_3d(frame, r, out, name, zi, band)
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(_to_u8(res["proj"]), cmap="gray"); ax[0].set_title("Композит (max по полосам)")
    ax[1].imshow(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB))
    ax[1].set_title("Дефекты (красн=тёмн, жёлт=бел)")
    pm = np.where(res["pmap"] > 0, res["pmap"], np.nan)
    im = ax[2].imshow(pm, cmap="hot", vmin=0, vmax=60); ax[2].set_title("Отклонение p%")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_result.png"), dpi=110)
    plt.close(fig)


def print_table(name: str, rows: list[dict]) -> None:
    print(f"\n==== {name} ====")
    print(" зона  x    тип полоса   медиана  тёмн.px  p20-50  p>50  бел.px  нас.  вердикт  причина")
    for r in rows:
        print(" z%-2d  %4d  %-2s %-8s %7.0f  %7d  %6d  %4d  %6d  %.2f  %-8s %s" % (
            r["zone"], r["x"], r.get("type", "?"), r["band"], r["median"], r["dark_px"],
            r["p20"], r["p50"], r["white_px"], r["sat"], r["verdict"], r.get("reason", "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = os.path.basename(args.specimen.rstrip("/\\"))
    out = args.out or os.path.join("results", "v2_pipeline", name)
    res = run(args.specimen, k=args.k)
    save(res, out, name)
    print_table(name, res["rows"])
    print(f"[ok] -> {out}")


if __name__ == "__main__":
    main()
