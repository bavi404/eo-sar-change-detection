import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ChangeDataset, get_eval_transforms, get_train_transforms, input_channel_count
from models import build_model
from utils import batch_iou_from_logits, build_loss
from utils.config_paths import resolve_path_value


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    accumulation_steps: int = 1,
) -> float:
    model.train()
    running_loss = 0.0
    accumulation_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    step = -1

    for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.unsqueeze(1).float().to(device, non_blocking=True)

        preds = model(images)
        loss = criterion(preds, masks)
        (loss / accumulation_steps).backward()
        running_loss += float(loss.detach().item())

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if step >= 0 and (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_iou(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    iou_sum = 0.0

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.unsqueeze(1).float().to(device, non_blocking=True)

        preds = model(images)
        iou_sum += batch_iou_from_logits(preds, masks, threshold=threshold)

    return iou_sum / max(len(loader), 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EO+SAR change detection (train+val only; never test).")
    default_cfg = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to config.yaml (default: next to train.py).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    dataset_cfg = cfg.get("dataset", {})
    concat_diff = bool(dataset_cfg.get("concat_diff", True))

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    train_root = resolve_path_value(data_cfg["train_root"], config_dir)
    val_root = resolve_path_value(data_cfg["val_root"], config_dir)
    output_dir = resolve_path_value(train_cfg["output_dir"], config_dir)

    print(f"Config: {config_path}")
    print(f"Train: {train_root}")
    print(f"Val:   {val_root}")
    print(f"Out:   {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Protocol: train on train split, validate on val split, do NOT use test during training.")
    eff_bs = int(train_cfg["batch_size"]) * int(train_cfg.get("accumulation_steps", 1))
    print(
        f"Batch: {train_cfg['batch_size']} x accumulation {train_cfg.get('accumulation_steps', 1)} "
        f"(effective optimizer step batch ~ {eff_bs})"
    )

    expected_ch = input_channel_count(concat_diff=concat_diff)
    if int(model_cfg["in_channels"]) != expected_ch:
        raise ValueError(
            f"model.in_channels={model_cfg['in_channels']} but dataset concat_diff={concat_diff} "
            f"expects {expected_ch} channels."
        )

    train_ds = ChangeDataset(
        split_root=train_root,
        transforms=get_train_transforms(crop_size=train_cfg["crop_size"]),
        concat_diff=concat_diff,
    )
    val_ds = ChangeDataset(
        split_root=val_root,
        transforms=get_eval_transforms(crop_size=train_cfg["crop_size"]),
        concat_diff=concat_diff,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=pin,
    )

    dec_att = model_cfg.get("decoder_attention_type")
    if dec_att is not None and isinstance(dec_att, str) and dec_att.lower() in {"none", "null", ""}:
        dec_att = None

    model = build_model(
        architecture=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
        decoder_attention_type=dec_att,
    ).to(device)

    loss_name = loss_cfg.get("name") or loss_cfg.get("type") or "bce_dice"
    criterion = build_loss(
        loss_name=str(loss_name),
        pos_weight=float(loss_cfg["pos_weight"]),
    )
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if weight_decay > 0:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"Optimizer: AdamW (weight_decay={weight_decay})")
    else:
        optimizer = Adam(model.parameters(), lr=lr)
        print("Optimizer: Adam")

    sched_cfg = train_cfg.get("lr_scheduler") or {}
    scheduler = None
    if bool(sched_cfg.get("enabled", False)):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=float(sched_cfg.get("min_lr", 1e-7)),
        )
        print(
            f"LR scheduler: ReduceLROnPlateau on val IoU "
            f"(factor={sched_cfg.get('factor', 0.5)}, patience={sched_cfg.get('patience', 3)})"
        )

    early_patience = int(train_cfg.get("early_stopping_patience", 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_model.pt"

    best_val_iou = -1.0
    epochs = int(train_cfg["epochs"])
    accum = int(train_cfg.get("accumulation_steps", 1))
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            accumulation_steps=accum,
        )
        val_iou = validate_iou(model, val_loader, device, threshold=train_cfg["threshold"])

        if scheduler is not None:
            scheduler.step(val_iou)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_iou={val_iou:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_iou > best_val_iou + 1e-8:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_ckpt)
            print(f"Saved best checkpoint to: {best_ckpt}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_patience > 0 and epochs_no_improve >= early_patience:
            print(
                f"Early stopping: no val IoU improvement for {early_patience} epochs "
                f"(stopped at epoch {epoch + 1}/{epochs})."
            )
            break

    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
