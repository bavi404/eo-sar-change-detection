import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    in_channels: int = 8,
    classes: int = 1,
    decoder_attention_type: str | None = "scse",
) -> nn.Module:
    arch = architecture.lower()

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
        )

    if arch in {"unetplusplus", "unet++"}:
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")
