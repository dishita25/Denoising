import torch
import torch.optim as optim
import numpy as np
from src.loss import loss_func, mse
from model.ZSN2N import network
from dataloader.PolyU import evaluate_polyu
from dataloader.CBSD68 import evaluate_artificial

def train_model(
    clean_img,
    noisy_img,
    n_chan=None,
    max_epoch=5000,
    lr=0.001,
    step_size=1000,
    gamma=0.5,
    device="cuda"
):

    if n_chan is None:
        n_chan = clean_img.shape[1]

    model = network(n_chan).to(device)
    print("The number of parameters of the network is:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    for epoch in range(1, max_epoch + 1):
        # --- Train step ---
        loss = loss_func(noisy_img, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Test step --- 
        with torch.no_grad():
            pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
            MSE = mse(clean_img, pred).item()
            PSNR = 10 * np.log10(1 / MSE)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{max_epoch}] | Loss: {loss.item():.6f} | PSNR: {PSNR:.2f} dB")

    return model


def test_model(model, dataset_name, dataset_path=None, device="cuda"):
    model.eval()

    if dataset_name == "polyu":
        print("Evaluating on PolyU...")
        results = evaluate_polyu(model, dataset_path, device)

    elif dataset_name == "mcmaster" or dataset_name == "cbsd68":
        print("Evaluating on Artificial dataset...")
        results = evaluate_artificial(model, dataset_path, device)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return results