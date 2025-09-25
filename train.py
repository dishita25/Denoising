import argparse
from src.trainer import train_model
from PIL import Image
from trainer import test_model

def main():
    parser = argparse.ArgumentParser(description="Train ZSN2N model with hyperparameters")

    # Hyperparameters
    parser.add_argument("--max_epoch", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=1000, help="LR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_chan", type=int, default=48, help="Number of channels (auto if None)")
    parser.add_argument("--noisy_img", type=str, default=None, help="Noisy Image")
    parser.add_argument("--clean_img", type=str, default=None, help="Clean Image")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset Name")

    args = parser.parse_args()

    noisy_img = Image.open(args.noisy_img).convert("RGB")
    clean_img = Image.open(args.clean_img).convert("RGB")

    # --- Training ---
    model = train_model(
        clean_img=clean_img,
        noisy_img=noisy_img,
        n_chan=args.n_chan,
        max_epoch=args.max_epoch,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        device=args.device
    )

    results = test_model(model, args.dataset)

    print("Test Results:", results)

if __name__ == "__main__":
    main()