import os
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import einops


# -------------------------------
parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--dataset', default='kodak', type=str, help='Dataset name')
parser.add_argument('--save', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=16, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=8, type=int, help='Number of pixels in pixel bank to use for training')
parser.add_argument('--nl', default=0.2, type=float, help='Noise level, for saltpepper and impulse noise, enter half the noise level.')
parser.add_argument('--nt', default='bernoulli', type=str, help='Noise type: gauss, poiss, saltpepper, bernoulli, impulse')
parser.add_argument('--loss', default='L1', type=str, help='Loss function type')
parser.add_argument('--train_image_idx', default=0, type=int, help='Index of image to train on')
args = parser.parse_args()


# -------------------------------
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0"

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
noise_level = args.nl
noise_type = args.nt
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])


# -------------------------------
# Function to add noise to an image
# -------------------------------
def add_noise(x, noise_level):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level
    elif noise_type == 'saltpepper':
        prob = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = 0
        noisy[prob > 1 - noise_level] = 1
    elif noise_type == 'bernoulli':
        prob = torch.rand_like(x)
        mask = (prob > noise_level).float()
        noisy = x * mask
    elif noise_type == 'impulse':
        prob = torch.rand_like(x)
        noise = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = noise[prob < noise_level]
    else:
        raise ValueError("Unsupported noise type")
    return noisy



# -------------------------------
def construct_pixel_bank():
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64  # Block size for processing

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()

        # Load image and add noise
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
        img = add_noise(img, noise_level).squeeze(0)
        img = img.cuda()[None, ...]  # Shape: [1, C, H, W]

        # Pad image and extract patches
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
        H_new = img.shape[-2] + WINDOW_SIZE
        W_new = img.shape[-1] + WINDOW_SIZE
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

        num_blk_w = img.shape[-1] // blk_sz
        num_blk_h = img.shape[-2] // blk_sz
        is_window_size_even = (WINDOW_SIZE % 2 == 0)
        topk_list = []

        # Iterate over blocks in the image
        for blk_i in range(num_blk_w):
            for blk_j in range(num_blk_h):
                start_h = blk_j * blk_sz
                end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
                start_w = blk_i * blk_sz
                end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

                sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
                sub_img_shape = sub_img_uf.shape

                if is_window_size_even:
                    sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
                else:
                    sub_img_uf_inp = sub_img_uf

                patch_windows = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1)
                patch_windows = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                    h=blk_sz, w=blk_sz
                )

                img_center = einops.rearrange(
                    sub_img_uf,
                    'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE,
                    h=sub_img_shape[-2], w=sub_img_shape[-1]
                )
                img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]

                if args.loss == 'L2':
                    distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
                elif args.loss == 'L1':
                    distance = torch.sum(torch.abs(img_center - patch_windows), dim=1)
                else:
                    raise ValueError(f"Unsupported loss type: {loss_type}")

                _, sort_indices = torch.topk(
                    distance,
                    k=NUM_NEIGHBORS,
                    largest=False,
                    sorted=True,
                    dim=-3
                )

                patch_windows_reshape = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
                topk_list.append(topk)

        # Merge the results from all blocks to form the pixel bank
        topk = torch.cat(topk_list, dim=0)
        topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
        topk = topk.permute(2, 3, 0, 1)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")

        file_name_without_ext = os.path.splitext(image_file)[0]
        np.save(os.path.join(bank_dir, file_name_without_ext), topk.cpu())

    print("Pixel bank construction completed for all images.")

# -------------------------------
class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.conv3(x)
        return x  # Return noise prediction (residual)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)

def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

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


def loss_func(img1, img2, model):
    pred1 = model(img1)
    loss = mse_loss(img2, pred1)
    return loss

def loss_func(img1, img2, model):
    noisy11, noisy12 = pair_downsampler(img1)
    noisy21, noisy22 = pair_downsampler(img2)

    pred11 = noisy11 - model(noisy11)
    pred12 = noisy12 - model(noisy12)

    pred21 = noisy21 - model(noisy21)
    pred22 = noisy22 - model(noisy22)

    loss_res1 = 0.5 * (mse_loss(noisy11, pred12) + mse_loss(noisy12, pred11))
    loss_res2 = 0.5 * (mse_loss(noisy21, pred22) + mse_loss(noisy22, pred21))

    loss_res = loss_res1 + loss_res2 # loss 1 

    noisy_denoised1 = img1 - model(img1)
    noisy_denoised2 = img2 - model(img2)

    denoised11, denoised12 = pair_downsampler(noisy_denoised1)
    denoised21, denoised22 = pair_downsampler(noisy_denoised2)

    # loss_cons1 = 0.5 * (mse_loss(pred11, img1) + mse_loss(pred2, img2)) # this can be pred, img1 and pred, img2 ( could work) - try this next
    loss_cons1 = 0.5 * (mse_loss(pred11, denoised11) + mse_loss(pred12, denoised12))
    loss_cons2 = 0.5 * (mse_loss(pred21, denoised21) + mse_loss(pred21, denoised22))

    loss_cons = loss_cons1 + loss_cons2 # loss 2 

    loss = loss_res + loss_cons # loss 2

    return loss

# -------------------------------
def train(model, optimizer, img_bank):
    N, H, W, C = img_bank.shape
    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Result shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(img1, img2, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        mse_val = mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred


# -------------------------------
def train_model():
    """Train model on one image's pixel bank"""
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))
    
    # Get the training image
    train_image_file = image_files[args.train_image_idx]
    print(f"\nTraining on image: {train_image_file}")
    
    # Load pixel bank for training image
    bank_path = os.path.join(bank_dir, os.path.splitext(train_image_file)[0])
    if not os.path.exists(bank_path + '.npy'):
        print(f"Pixel bank for {train_image_file} not found!")
        return None
    
    img_bank_arr = np.load(bank_path + '.npy')
    if img_bank_arr.ndim == 3:
        img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
    img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
    
    if noise_type=='gauss' and noise_level==10 or noise_type=='bernoulli':
        args.mm=2
    elif noise_type=='gauss' and noise_level==25:
        args.mm = 4
    else:
        args.mm = 8
    img_bank = img_bank[:args.mm]
    
    img_bank = torch.from_numpy(img_bank).to(device)
    
    # Load clean image to determine number of channels
    image_path = os.path.join(image_folder, train_image_file)
    clean_img = Image.open(image_path)
    clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
    n_chan = clean_img_tensor.shape[1]
    
    # Initialize model
    model = Network(n_chan).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training setup
    max_epoch = 3000
    lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)
    
    # Train
    print("Training model...")
    for epoch in range(max_epoch):
        loss = train(model, optimizer, img_bank)
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{max_epoch}, Loss: {loss:.6f}")
    
    print("Training completed!")
    return model


def inference_on_dataset(model):
    """Run inference on entire dataset using trained model"""
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    os.makedirs(args.out_image, exist_ok=True)

    avg_PSNR = 0
    avg_SSIM = 0

    print("\nRunning inference on entire dataset...")
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        bank_path = os.path.join(bank_dir, os.path.splitext(image_file)[0])
        if not os.path.exists(bank_path + '.npy'):
            print(f"Pixel bank for {image_file} not found, skipping.")
            continue

        img_bank_arr = np.load(bank_path + '.npy')
        if img_bank_arr.ndim == 3:
            img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
        img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
        img_bank = torch.from_numpy(img_bank).to(device)

        noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

        PSNR, out_img = test(model, noisy_img, clean_img_tensor)
        out_img_pil = to_pil_image(out_img.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)

        noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, channel_axis=2)
        print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    avg_PSNR /= len(image_files)
    avg_SSIM /= len(image_files)
    print(f"\nAverage PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")


# -------------------------------
if __name__ == "__main__":
    print("Constructing pixel banks ...")
    construct_pixel_bank()
    print("\n" + "="*50)
    print("Training model on single image...")
    print("="*50)
    trained_model = train_model()
    if trained_model is not None:
        print("\n" + "="*50)
        print("Running inference on entire dataset...")
        print("="*50)
        inference_on_dataset(trained_model)