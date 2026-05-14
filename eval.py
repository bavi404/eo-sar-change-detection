import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ChangeDataset, get_eval_transforms, input_channel_count
from models import build_model
from utils.config_paths import resolve_path_value


@torch.no_grad()
def forward_mean_probs(model: torch.nn.Module, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Sigmoid probs; optional TTA = average with H/V flips (eval-only, no label leakage)."""
    logits = model(images)
    probs = torch.sigmoid(logits)
    if not use_tta:
        return probs
    ph = torch.flip(torch.sigmoid(model(torch.flip(images, [-1]))), [-1])
    pv = torch.flip(torch.sigmoid(model(torch.flip(images, [-2]))), [-2])
    return (probs + ph + pv) / 3.0


def dataset_mask_value_summary(dataset: ChangeDataset, max_items: int = 200) -> list[int]:
    values = set()
    n = min(len(dataset), max_items)
    for i in range(n):
        _, mask = dataset[i]
        uniq = torch.unique(mask)
        values.update(int(v.item()) for v in uniq)
        if values == {0, 1}:
            break
    return sorted(values)


@torch.no_grad()
def evaluate_split(
    model,
    loader,
    device,
    threshold: float = 0.5,
    *,
    use_tta: bool = False,
) -> dict:
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        probs = forward_mean_probs(model, images, use_tta)
        preds = (probs >= threshold).long().squeeze(1)
        if preds.shape != masks.shape:
            raise ValueError(f"Shape mismatch: pred {preds.shape} vs mask {masks.shape}")

        tp += int(((preds == 1) & (masks == 1)).sum().item())
        fp += int(((preds == 1) & (masks == 0)).sum().item())
        fn += int(((preds == 0) & (masks == 1)).sum().item())
        tn += int(((preds == 0) & (masks == 0)).sum().item())

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def build_loader(
    split_root: str,
    batch_size: int,
    num_workers: int,
    crop_size: int,
    *,
    concat_diff: bool,
) -> DataLoader:
    dataset = ChangeDataset(
        split_root=split_root,
        transforms=get_eval_transforms(crop_size=crop_size),
        concat_diff=concat_diff,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def print_metrics(name: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n{name} metrics")
    print(f"IoU:       {metrics['iou']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1:        {metrics['f1']:.6f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"[[{cm['tn']}, {cm['fp']}], [{cm['fn']}, {cm['tp']}]]")


@torch.no_grad()
def tune_threshold_on_validation(
    model,
    val_loader: DataLoader,
    device,
    thresholds: list[float],
    *,
    use_tta: bool = False,
) -> tuple[float, dict]:
    best_t = thresholds[0]
    best_metrics: dict | None = None
    best_iou = -1.0

    for t in thresholds:
        metrics = evaluate_split(model, val_loader, device, threshold=float(t), use_tta=use_tta)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_t = float(t)
            best_metrics = metrics

    assert best_metrics is not None
    return best_t, best_metrics


@torch.no_grad()
def save_prediction_examples(
    model,
    dataset: ChangeDataset,
    device,
    output_dir: Path,
    threshold: float = 0.5,
    num_examples: int = 5,
    *,
    use_tta: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(num_examples, len(dataset))

    model.eval()
    for i in range(n):
        image, mask = dataset[i]
        input_batch = image.unsqueeze(0).to(device)
        probs = forward_mean_probs(model, input_batch, use_tta)
        pred = (probs >= threshold).float().squeeze().cpu().numpy()

        image_np = image.cpu().numpy()  # (C, H, W)
        mask_np = mask.cpu().numpy()

        eo_pre = image_np[0:3].transpose(1, 2, 0)
        eo_post = image_np[3:6].transpose(1, 2, 0)
        channels = image_np.shape[0]

        if channels >= 12:
            eo_diff_vis = image_np[8:11].transpose(1, 2, 0)
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(eo_pre)
            axes[0].set_title("Input EO Pre")
            axes[0].axis("off")

            axes[1].imshow(eo_post)
            axes[1].set_title("Input EO Post")
            axes[1].axis("off")

            axes[2].imshow(eo_diff_vis)
            axes[2].set_title("EO abs(post - pre)")
            axes[2].axis("off")

            axes[3].imshow(mask_np, cmap="gray")
            axes[3].set_title("Ground Truth")
            axes[3].axis("off")

            axes[4].imshow(pred, cmap="gray")
            axes[4].set_title("Prediction")
            axes[4].axis("off")
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(eo_pre)
            axes[0].set_title("Input EO Pre")
            axes[0].axis("off")

            axes[1].imshow(eo_post)
            axes[1].set_title("Input EO Post")
            axes[1].axis("off")

            axes[2].imshow(mask_np, cmap="gray")
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

            axes[3].imshow(pred, cmap="gray")
            axes[3].set_title("Prediction")
            axes[3].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"example_{i:02d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved {n} visualization examples to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on val+test; threshold tune on val only.")
    default_cfg = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to config.yaml (default: next to eval.py).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    concat_diff = bool(dataset_cfg.get("concat_diff", True))
    expected_ch = input_channel_count(concat_diff=concat_diff)

    val_root = resolve_path_value(data_cfg["val_root"], config_dir)
    test_root = resolve_path_value(data_cfg["test_root"], config_dir)
    out_dir = resolve_path_value(train_cfg["output_dir"], config_dir)

    print(f"Config: {config_path}")
    print(f"Val:   {val_root}")
    print(f"Test:  {test_root}")
    print(f"Out:   {out_dir}")
    print("Protocol: evaluate on provided test split only after training/validation are finalized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_tta = bool(eval_cfg.get("use_tta", False))
    if use_tta:
        print("TTA enabled: averaging probs with horizontal + vertical flips (eval only).")

    if int(model_cfg["in_channels"]) != expected_ch:
        raise ValueError(
            f"model.in_channels={model_cfg['in_channels']} but concat_diff={concat_diff} "
            f"implies {expected_ch} channels."
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

    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {ckpt_path}")

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    crop_size = int(train_cfg["crop_size"])

    val_loader = build_loader(
        str(val_root), batch_size, num_workers, crop_size, concat_diff=concat_diff
    )
    test_loader = build_loader(
        str(test_root), batch_size, num_workers, crop_size, concat_diff=concat_diff
    )
    test_dataset_for_vis = ChangeDataset(
        split_root=test_root,
        transforms=get_eval_transforms(crop_size=crop_size),
        concat_diff=concat_diff,
    )
    val_mask_values = dataset_mask_value_summary(val_loader.dataset)
    test_mask_values = dataset_mask_value_summary(test_loader.dataset)
    print(f"Val mask unique values after remap (sampled): {val_mask_values}")
    print(f"Test mask unique values after remap (sampled): {test_mask_values}")

    if bool(eval_cfg.get("tune_threshold_on_val", False)):
        grid = eval_cfg.get("threshold_grid")
        if grid is not None:
            thresh_candidates = [float(x) for x in grid]
        else:
            thresh_candidates = [round(float(t), 4) for t in torch.linspace(0.05, 0.95, 19).tolist()]

        best_t, tune_val_metrics = tune_threshold_on_validation(
            model,
            val_loader,
            device,
            thresholds=thresh_candidates,
            use_tta=use_tta,
        )
        threshold = best_t

        thresh_path = out_dir / "best_threshold.json"
        with thresh_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": threshold,
                    "use_tta": use_tta,
                    "val_iou_at_selected_threshold": tune_val_metrics["iou"],
                    "val_precision": tune_val_metrics["precision"],
                    "val_recall": tune_val_metrics["recall"],
                    "val_f1": tune_val_metrics["f1"],
                    "threshold_grid": thresh_candidates,
                },
                f,
                indent=2,
            )
        print(f"Threshold tuned on validation only: {threshold:.4f} (saved {thresh_path})")
    else:
        threshold = float(train_cfg["threshold"])
        print(
            f"Fixed threshold from config (tune_threshold_on_val=false): {threshold:.4f}. "
            "TTA is controlled by evaluation.use_tta in eval.py."
        )

    val_metrics = evaluate_split(model, val_loader, device, threshold=threshold, use_tta=use_tta)
    test_metrics = evaluate_split(model, test_loader, device, threshold=threshold, use_tta=use_tta)

    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    vis_dir = out_dir / "visualizations"
    save_prediction_examples(
        model=model,
        dataset=test_dataset_for_vis,
        device=device,
        output_dir=vis_dir,
        threshold=threshold,
        num_examples=5,
        use_tta=use_tta,
    )


if __name__ == "__main__":
    main()
