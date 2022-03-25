from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING
from TTS.enhancer.models.bwe import BWEArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class BaseEnhancerConfig(BaseTrainingConfig):
    """Defines parameters for a Generic Encoder model."""

    model_args: BWEArgs = field(default_factory=BWEArgs)
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])

    # model params
    audio_augmentation: Dict = field(default_factory=lambda: {})