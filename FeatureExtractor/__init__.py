from .embedding import Embedding
from .extractors import (
    CLIP,
    ContinuousFlatten,
    DiscreteFlatten,
    FeatureExtractor,
    Flatten,
    RandomConvNet,
    RandomVGG,
    SpatialStatistics,
    VGG16,
)

RandomConvNetExtractor = RandomConvNet
FlattenSpatialExtractor = DiscreteFlatten
CLIPExtractor = CLIP

__all__ = [
    "Embedding",
    "FeatureExtractor",
    "DiscreteFlatten",
    "ContinuousFlatten",
    "Flatten",
    "SpatialStatistics",
    "RandomConvNet",
    "RandomVGG",
    "VGG16",
    "CLIP",
    "FlattenSpatialExtractor",
    "RandomConvNetExtractor",
    "CLIPExtractor",
]
