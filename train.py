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
    max_epoch = trial.suggest_int("max_epoch", 4000, 5500, step=500) if trial else args.max_epoch #removing 6000
    lr = trial.suggest_categorical("lr", [0.001, 0.01, 0.05]) if trial else args.lr # removing 0.1
    step_size = trial.suggest_int("step_size", 500, 2000, step=500) if trial else args.step_size
    mask_ratio = trial.suggest_categorical("mask_ratio", [0.5, 0.55, 0.6, 0.65]) if trial else args.mask_ratio
    blind_spot_weight = trial.suggest_categorical("blind_spot_weight", [0.25, 0.5, 0.75, 1]) if trial else args.blind_spot_weight
    # gamma = trial.suggest_categorical("gamma", [0.5, 0.6, 0.7]) if trial else args.gamma
    gamma = args.gamma
    n_chan = args.n_chan   

    # --- Load and preprocess images ---
    # clean_img = Image.open(args.clean_img).convert("RGB")
    # if args.dataset in ["Mcmaster", "CBSD", "kodak"]:
    #     noisy_img = Image.open(args.noisy_img).convert("RGB")
    # else:
    #     noisy_img = Image.open(args.noisy_img).convert("RGB")

    # center_crop = T.CenterCrop((256, 256))
    # clean_img = to_tensor(center_crop(clean_img)).unsqueeze(0)
    # noisy_img = to_tensor(center_crop(noisy_img)).unsqueeze(0)


    # --- Training ---
    model = train_model(
        noisy_dir = args.noisy_dir,
        #clean_img=clean_img,
        #noisy_img=noisy_img,
        n_chan=n_chan,
        max_epoch=max_epoch,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        mask_ratio=mask_ratio,
        blind_spot_weight=blind_spot_weight,
        batch_size = args.batch_size,
        device=args.device
    )


    # --- Testing ---
    psnr, ssim = test_model(model, args.dataset, args.dataset_path, device=args.device, noise_level=args.noise_level)
    return psnr, ssim



def objective(trial, args):
    avg_psnr, avg_ssim = run_training(args, trial)
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Train ZSN2N model with hyperparameters")


    # Hyperparameters
    parser.add_argument("--max_epoch", type=int, default=4500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=1500, help="LR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_chan", type=int, default=3, help="Number of channels (auto if None)")
    #parser.add_argument("--noisy_img", type=str, default="/kaggle/input/original/Canon5D2_bag_Real.JPG", help="Noisy image path")
    #parser.add_argument("--clean_img", type=str, default="/kaggle/input/original/Canon5D2_bag_mean.JPG", help="Path to clean image")
    parser.add_argument("--dataset", type=str, default="polyu", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/kaggle/input/polyucropped", help="Dataset path for inference")
    parser.add_argument("--noise_level", type=int, default=25, help="Noise Level")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Mask ratio")
    parser.add_argument("--blind_spot_weight", type=float, default=1, help="blind_spot_weight")
    parser.add_argument("--noisy_dir", type=str, default="/kaggle/input/original/Canon5D2_bag_Real.JPG", help="Noisy images director")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")




    args = parser.parse_args()


    if args.optuna:
        print("Using Optuna for hyperparameter optimization...")
        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(lambda trial: objective(trial, args), n_trials=500)

        print("Best trials:")
        for t in study.best_trials:
            print(f"  Values (PSNR, SSIM): {t.values}")
            print("  Params: ")
            for key, value in t.params.items():
                print(f"    {key}: {value}")

    else:
        print("Direct Training")
        psnr, ssim = run_training(args)
        print(f"Test Results: PSNR:{psnr}, SSIM: {ssim}")


if __name__ == "__main__":
    main()