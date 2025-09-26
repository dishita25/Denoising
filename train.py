import argparse
from src.trainer import train_model, test_model
from PIL import Image
import torchvision.transforms as T
from src.utils import add_noise
import optuna


# transforms
to_tensor = T.ToTensor()


def run_training(args, trial=None):
    # --- Hyperparameters (Optuna overrides if trial is provided) ---
    max_epoch = trial.suggest_int("max_epoch", 3000, 6000, step=500) if trial else args.max_epoch
    lr = trial.suggest_categorical("lr", [0.001, 0.01, 0.05, 0.1]) if trial else args.lr
    step_size = trial.suggest_int("step_size", 500, 2000, step=500) if trial else args.step_size
    mask_ratio = trial.suggest_categorical("mask_ratio", [0.5, 0.55, 0.6, 0.65]) if trial else args.mask_ratio
    gamma = trial.suggest_categorical("gamma", [0.5, 0.6, 0.7]) if trial else args.gamma
    n_chan = args.n_chan   

    # --- Load and preprocess images ---
    clean_img = Image.open(args.clean_img).convert("RGB")
    if args.dataset in ["Mcmaster", "CBSD", "kodak"]:
        noisy_img = Image.open(args.noisy_img).convert("RGB")
    else:
        noisy_img = Image.open(args.noisy_img).convert("RGB")

    center_crop = T.CenterCrop((256, 256))
    clean_img = to_tensor(center_crop(clean_img)).unsqueeze(0)
    noisy_img = to_tensor(center_crop(noisy_img)).unsqueeze(0)


    # --- Training ---
    model = train_model(
        clean_img=clean_img,
        noisy_img=noisy_img,
        n_chan=n_chan,
        max_epoch=max_epoch,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        mask_ratio=mask_ratio,
        device=args.device
    )


    # --- Testing ---
    results = test_model(model, args.dataset, args.dataset_path, device=args.device, noise_level=args.noise_level)
    return results



def objective(trial, args):
    avg_psnr = run_training(args, trial)
    return avg_psnr


def main():
    parser = argparse.ArgumentParser(description="Train ZSN2N model with hyperparameters")


    # Hyperparameters
    parser.add_argument("--max_epoch", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=1000, help="LR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_chan", type=int, default=3, help="Number of channels (auto if None)")
    parser.add_argument("--noisy_img", type=str, default="/kaggle/input/original/Canon5D2_bag_Real.JPG", help="Noisy image path")
    parser.add_argument("--clean_img", type=str, default="/kaggle/input/original/Canon5D2_bag_mean.JPG", help="Path to clean image")
    parser.add_argument("--dataset", type=str, default="polyu", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/kaggle/input/polyucropped", help="Dataset path for inference")
    parser.add_argument("--noise_level", type=int, default=25, help="Noise Level")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Mask ratio")



    args = parser.parse_args()


    if args.optuna:
        print("Using Optuna for hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args), n_trials=20)

        print("Best trial:")
        print(f"  Value (PSNR): {study.best_trial.value:.2f}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("Direct Training")
        results = run_training(args)
        print("Test Results:", results)


if __name__ == "__main__":
    main()