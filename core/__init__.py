"""
Основные модули обработки изображений и детекции
"""

from .image_processing import (
    apply_histogram_normalization,
    preprocess_for_detection,
    resize_with_letterbox,
    restore_mask_letterbox,
    convert_to_grayscale,
    prepare_for_display,
    prepare_zone_tiles,
    stitch_tile_probs,
)

from .zone_detection import (
    find_top_zones,
    find_top_zones_improved,
    extract_zone_roi,
    analyze_zones,
    apply_image_correction,
    pixels_to_mm,
    get_zone_size_mm
)

from .defect_detection import (
    predict_defects,
    predict_defects_batch,
    predict_defects_batch_probs,
    refine_defect_mask,
    classify_defects_gost,
    apply_defects_overlay,
    draw_zones,
    calculate_defect_statistics,
)

from .cameras import (
    CameraCapture,
    CameraManager,
    GigECameraCapture,
    GigECameraManager,
    HARVESTER_AVAILABLE,
    PylonCameraCapture,
    PylonCameraManager,
    PYLON_AVAILABLE,
    EbusCameraCapture,
    EbusCameraManager,
    EBUS_AVAILABLE,
    init_ebus_sdk,
    bitmap_to_numpy,
    log_ebus,
)

__all__ = [
    'apply_histogram_normalization',
    'preprocess_for_detection',
    'resize_with_letterbox',
    'restore_mask_letterbox',
    'convert_to_grayscale',
    'prepare_for_display',
    'prepare_zone_tiles',
    'stitch_tile_probs',
    'find_top_zones',
    'find_top_zones_improved',
    'extract_zone_roi',
    'analyze_zones',
    'apply_image_correction',
    'pixels_to_mm',
    'get_zone_size_mm',
    'predict_defects',
    'predict_defects_batch',
    'predict_defects_batch_probs',
    'refine_defect_mask',
    'classify_defects_gost',
    'apply_defects_overlay',
    'draw_zones',
    'calculate_defect_statistics',
    'CameraCapture',
    'CameraManager',
    'GigECameraCapture',
    'GigECameraManager',
    'HARVESTER_AVAILABLE',
    'PylonCameraCapture',
    'PylonCameraManager',
    'PYLON_AVAILABLE',
    'EbusCameraCapture',
    'EbusCameraManager',
    'EBUS_AVAILABLE',
    'init_ebus_sdk',
    'bitmap_to_numpy',
    'log_ebus',
]