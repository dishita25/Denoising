import torch.optim as optim
from src.loss import loss_func
from src.model.ZSN2N import network
from src.dataloader.PolyU import evaluate_polyu
from src.dataloader.CBSD68 import evaluate_artificial
from src.dataloader.noisy_dataset import NoisyImageDataset
from src.utils import test
import torchvision.transforms as T 
from torch.utils.data import DataLoader


def train_model(
    noisy_dir,
    #clean_img,
    #noisy_img,
    n_chan,
    max_epoch,
    lr,
    step_size,
    gamma,
    mask_ratio,
    blind_spot_weight,
    device="cuda",
):

    if n_chan is None:
        n_chan = clean_img.shape[1]

    model = network(n_chan).to(device)
    print("The number of parameters of the network is:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    #clean_img = clean_img.to(device)
    #noisy_img = noisy_img.to(device)

    transform = T.Compose([
        T.CenterCrop((256,256)),
        T.toTensor()
    ])

    train_dataset = NoisyImageDataset(noisy_dir=noisy_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # for epoch in range(1, max_epoch + 1):
    #     # --- Train step ---
    #     loss = loss_func(noisy_img, model, mask_ratio=mask_ratio, blind_spot_weight=blind_spot_weight)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()

    #     # --- Test step --- 
    #     PSNR, ssim = test(model, noisy_img, clean_img)

    #     if epoch % 100 == 0 or epoch == 1:
    #         print(f"Epoch [{epoch}/{max_epoch}] | Loss: {loss.item():.6f} | PSNR: {PSNR:.4f} dB | SSIM: {ssim:.4f}")

    # return model

    for epoch in range(1, max_epoch + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, noisy_imgs in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device)
            
            batch_loss = 0.0
            for i in range(noisy_imgs.size(0)):
                noisy_img = noisy_imgs[i:i+1]
                loss = loss_func(noisy_img, model, mask_ratio=mask_ratio, blind_spot_weight=blind_spot_weight)
                batch_loss += loss
            
            batch_loss = batch_loss / noisy_imgs.size(0)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{max_epoch}] | Loss: {epoch_loss:.6f}")

    return model


def test_model(model, dataset_name, dataset_path, device="cuda", noise_level = None):
    model.eval()

    transform = T.Compose([
        T.CenterCrop((256, 256)),
        T.ToTensor()
    ])

    if dataset_name == "polyu":
        print("Evaluating on PolyU...")
        psnr, ssim = evaluate_polyu(model, dataset_path, device)

    elif dataset_name == "Mcmaster" or dataset_name == "CBSD" or dataset_name == "kodak":
        print("Evaluating on Artificial dataset...")
        psnr, ssim = evaluate_artificial(model, dataset_name, noise_level, dataset_path, device)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return psnr, ssim