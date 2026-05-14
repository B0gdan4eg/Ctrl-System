"""
Модули для работы с камерами
"""

from .camera import (
    CameraCapture,
    CameraManager,
)

try:
    from .gige_camera import (
        GigECameraCapture,
        GigECameraManager,
        HARVESTER_AVAILABLE
    )
except ImportError:
    HARVESTER_AVAILABLE = False
    GigECameraCapture = None
    GigECameraManager = None

try:
    from .pylon_camera import (
        PylonCameraCapture,
        PylonCameraManager,
        PYLON_AVAILABLE,
    )
except ImportError:
    PYLON_AVAILABLE = False
    PylonCameraCapture = None
    PylonCameraManager = None

try:
    from .ebus_camera import (
        EbusCameraCapture,
        EbusCameraManager,
        EBUS_AVAILABLE,
        init_ebus_sdk,
        bitmap_to_numpy,
        log_ebus
    )
except ImportError:
    EBUS_AVAILABLE = False
    EbusCameraCapture = None
    EbusCameraManager = None
    init_ebus_sdk = None
    bitmap_to_numpy = None
    log_ebus = None

__all__ = [
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
