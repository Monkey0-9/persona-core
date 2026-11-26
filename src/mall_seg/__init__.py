"""Production Mall Customer Segmentation package."""

from mall_seg.config import AppConfig, get_config, load_config
from mall_seg.profiling import PersonaProfile, SegmentKPI, generate_segment_kpis
from mall_seg.schema import NUMERIC_FEATURE_COLUMNS, REQUIRED_COLUMNS, validate_raw_dataframe

__all__ = [
    "AppConfig",
    "NUMERIC_FEATURE_COLUMNS",
    "PersonaProfile",
    "REQUIRED_COLUMNS",
    "SegmentKPI",
    "generate_segment_kpis",
    "get_config",
    "load_config",
    "validate_raw_dataframe",
    "__version__",
]
__version__ = "1.0.0"
