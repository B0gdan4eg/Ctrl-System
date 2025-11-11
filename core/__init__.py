"""
Основные модули обработки изображений и детекции
"""

from .image_processing import (
    apply_histogram_normalization,
    preprocess_for_detection,
    convert_to_grayscale,
    prepare_for_display
)

from .zone_detection import (
    find_top_zones,
    extract_zone_roi
)

from .defect_detection import (
    predict_defects,
    visualize_defects,
    calculate_defect_statistics,
    create_defect_heatmap
)

__all__ = [
    'apply_histogram_normalization',
    'preprocess_for_detection',
    'convert_to_grayscale',
    'prepare_for_display',
    'find_top_zones',
    'extract_zone_roi',
    'predict_defects',
    'visualize_defects',
    'calculate_defect_statistics',
    'create_defect_heatmap'
]