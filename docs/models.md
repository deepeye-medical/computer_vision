# Models

Select a network with a second training YAML, for example
`--config configs/model_resnet.yaml`. Model settings use immutable Pydantic types
and reject unknown fields.

## Input modes and coordinates

CNN and ResNet kernel dimensions select 2D images or 3D volumes. With 2D kernels,
`slice_wise: true` processes `(batch, channels, depth, height, width)` inputs as
independent slices. Spatial pooling produces one feature vector per slice;
`slice_pooling` combines them before classification. Both pooling settings accept
`mean` and `mean_max`. With 3D kernels, convolutions combine neighboring slices.
The supplied loader and augmentation handle 2D RGB images. Other modes need
matching loader and augmentation code.

`use_coordinate_channels` adds signed grid coordinates in [-1, 1], in width,
height, then depth order. Slice mode adds them before splitting the volume.
Singleton axes receive zero. Coordinates describe grid position rather than
physical distance. `in_channels` counts stored channels only. Slice pooling
loses order unless the backbone uses depth coordinates.

## CNN and ResNet

`CNNStageConfig` controls channels, kernels, strides, padding, and pooling.
Each stage applies convolution, ReLU, BatchNorm, then optional max pooling.
The presets use three 32-channel stages, unpadded convolutions, spatial mean
pooling, and a 64-unit hidden classifier. The head has no BatchNorm, allowing
single-volume batches. The image preset requires at least 32 pixels per spatial
axis; the volume preset also needs at least 22 slices.

`StemConfig` and `ResStageConfig` control ResNet construction. A stage sets
channels, block count, kernels, and stride. The first block applies the stride
in both residual and skip branches. See `model_resnet.yaml` and
`model_resnet_3d.yaml` for the small image and volume defaults.

| Preset in `configs/` | Input | Weights |
| --- | --- | --- |
| `model_imagenet_resnet.yaml` / `model_resnet34_image.yaml` | Image | ImageNet ResNet-18 / 34 |
| `model_resnet18_slice.yaml` / `model_resnet34_slice.yaml` | Independent slices | ImageNet ResNet-18 / 34 |
| `model_resnet18_volume.yaml` / `model_resnet34_volume.yaml` | Volume | Inflated ImageNet ResNet-18 / 34 |
| `model_r2plus1d_18.yaml` | Volume | Kinetics R(2+1)D-18 |

## ViT

`PatchConfig` sets input and patch sizes; `EncoderConfig` sets layer count,
token width, attention heads, and MLP width. Two-axis patches encode images.
Slice mode encodes each slice independently before combining features.
Three-axis patches use attention across the whole volume. Input sizes must match
the configured grid and be divisible by patch sizes. Token pooling accepts
`cls`, `mean`, or `mean_max`; slice pooling accepts `mean` or `mean_max`.

Use `model_vit.yaml` for a small untrained model or `model_imagenet_vit.yaml`
for pretrained ViT-B/16. Slice and volume presets require custom loaders.
The volume preset uses `[32, 224, 224]` inputs and `[4, 16, 16]` patches, giving
1568 patch tokens. Attention memory grows with the square of the token count.
Slice mode accepts any depth with the configured height and width.

## Pretrained weights

ResNet and ViT train all parameters and have no backbone-freezing option.
`weights` selects a pinned source; null starts from random weights. Training
loads cached weights and downloads missing files. Checkpoint loading and resumes
restore saved weights without downloads. Direct Python callers must call
`load_pretrained_weights()` after construction. Constructors set normalization
without loading pretrained parameters.

ImageNet ResNet kernels repeat across depth and divide by the depth kernel size.
R(2+1)D keeps its spatial and depth filters. Non-RGB inputs receive the summed
RGB filters divided by the stored channel count; normalization uses the RGB
mean and standard deviation averaged across channels. Coordinate filters start
at zero. Stored intensities stay in [0, 1]; normalization runs inside the network.

ViT uses the same channel adaptation. Volume patch kernels repeat across depth
and divide by the depth patch size. Learned position embeddings resize spatially
and repeat across depth without scaling. The loader accepts the older torchvision
MLP parameter names, checks parameter shapes, and rejects unused source parameters.

`TorchvisionClassifier` provides separate 2D ResNet and ViT adapters. It can freeze
the backbone while training the task head.
