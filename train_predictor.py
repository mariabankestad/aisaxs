#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from aisaxs.models.mlps.dtc_model import DCTModel, make_dct2_ortho_uniform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DCTModel on concatenated train/val/test splits from the training dataset."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/lnp/setup_shell_shift/train.pt",
        help="Path to train split.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/lnp/setup_shell_shift/val.pt",
        help="Path to validation split.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/lnp/setup_shell_shift/test.pt",
        help="Path to internal test split that is also included in training.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint.pt",
        help="Checkpoint filename.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=1025,
        help="Hidden size for DCTModel.",
    )
    parser.add_argument(
        "--dct-size",
        type=int,
        default=300,
        help="DCT basis size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW (cosine schedule starts here).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Final learning rate at the end of the cosine schedule.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print training loss every N epochs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N epochs.",
    )
    return parser.parse_args()


def load_training_data(
    train_path: str,
    val_path: str,
    test_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.load(train_path)
    val = torch.load(val_path)
    test = torch.load(test_path)

    data = train["y"].float()
    data_val = val["y"].float()
    data_test = test["y"].float()

    x_train = train["x"].float()
    x_val = val["x"].float()
    x_test = test["x"].float()

    Y = torch.cat([data, data_val, data_test], dim=0)
    X = torch.cat([x_train, x_val, x_test], dim=0)

    return X, Y


def build_model(dct_size: int, hidden: int, device: str) -> DCTModel:
    Phi_dct = torch.from_numpy(make_dct2_ortho_uniform(dct_size, dct_size)).float()
    model = DCTModel(Phi_dct, 9, hidden=hidden).to(device)
    return model


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / args.checkpoint_name

    X, Y = load_training_data(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    dataset = TensorDataset(X, Y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    device = args.device
    model = build_model(
        dct_size=args.dct_size,
        hidden=args.hidden,
        device=device,
    )
    model.train()

    criterion = torch.nn.SmoothL1Loss()

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
        ]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr
    )

    for epoch in range(args.epochs):
        losses = []

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            min_, _ = y.min(dim=1, keepdim=True)
            max_, _ = y.max(dim=1, keepdim=True)
            dy = (max_ - min_).abs().detach()

            core_std, mu, sigma = model(x)
            y_hat = mu + sigma * core_std

            loss = criterion(y_hat / dy, y / dy) * dy.mean() * 10

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        loss_ = sum(losses) / len(losses)
        scheduler.step()

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | total {loss_:.4e} | lr {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()