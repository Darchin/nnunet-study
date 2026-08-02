from typing import Callable
import torch.nn as nn

type ModuleFactory = Callable[..., nn.Module]


type SomethingNd[T] = tuple[T, ...]
type Something1d[T] = tuple[T]
type Something2d[T] = tuple[T, T]
type Something3d[T] = tuple[T, T, T]

type ShapeNd = SomethingNd[int]
type Shape1d = Something1d[int]
type Shape2d = Something2d[int]
type Shape3d = Something3d[int]

type RatioNd = SomethingNd[float]
type Ratio1d = Something1d[float]
type Ratio2d = Something2d[float]
type Ratio3d = Something3d[float]
