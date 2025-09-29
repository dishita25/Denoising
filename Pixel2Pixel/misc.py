# def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
#     return nn.MSELoss()(gt, pred)

# def loss_func(noisy_img, model):
#     """
#     downsample, pass to model subtract, then do original - output, downsample ( I could try noise consistency as neoightbouting pixels shoufl have same noise )
#     """
#     # --- Original consistency losses ---
#     noisy1, noisy2 = pair_downsampler(noisy_img)
#     pred1 = noisy1 - model(noisy1)
#     pred2 = noisy2 - model(noisy2)
#     loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

#     noisy_denoised = noisy_img - model(noisy_img)
#     denoised1, denoised2 = pair_downsampler(noisy_denoised)
#     loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

#     loss = loss_res + loss_cons
#     return loss


# def loss_func(img1, img2, model):
#     pred1 = model(img1)
#     loss = mse_loss(img2, pred1)
#     return loss
