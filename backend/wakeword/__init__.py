"""Modular offline openWakeWord ONNX wake-word package for LocalVox."""

from backend.wakeword.detector import (
    OWW_SAMPLE_RATE,
    OWW_STEP_SAMPLES,
    OpenWakeWordEngine,
    WakeWordDetector,
)
from backend.wakeword.enrollment import (
    REQUIRED_ENROLLMENT_SAMPLES,
    EnrollmentSampleResult,
    PersonalizedVoiceProfile,
    VoiceProfileSummary,
)
from backend.wakeword.events import (
    WakeWordDetectionEvent,
    WakeWordEngineInfo,
    WakeWordError,
    WakeWordModelMissingError,
)
from backend.wakeword.models import (
    OPENWAKEWORD_ONNX_ARTIFACTS,
    check_wakeword_models,
    download_wakeword_models,
    ensure_wakeword_models_exist,
    wakeword_disk_size_mb,
)

__all__ = [
    "OPENWAKEWORD_ONNX_ARTIFACTS",
    "OWW_SAMPLE_RATE",
    "OWW_STEP_SAMPLES",
    "REQUIRED_ENROLLMENT_SAMPLES",
    "EnrollmentSampleResult",
    "OpenWakeWordEngine",
    "PersonalizedVoiceProfile",
    "VoiceProfileSummary",
    "WakeWordDetectionEvent",
    "WakeWordDetector",
    "WakeWordEngineInfo",
    "WakeWordError",
    "WakeWordModelMissingError",
    "check_wakeword_models",
    "download_wakeword_models",
    "ensure_wakeword_models_exist",
    "wakeword_disk_size_mb",
]
