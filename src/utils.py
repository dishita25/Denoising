import torch.nn.functional as F
import torch 

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

def apply_blindspot_mask(img, mask_ratio):
    """
    Randomly masks out pixels (blind-spot).
    img: [B, C, H, W]
    mask_ratio: fraction of pixels to mask
    """
    B, C, H, W = img.shape
    mask = torch.rand(B, 1, H, W, device=img.device) < mask_ratio  # True where masked
    noisy_masked = img.clone()
    noisy_masked[mask.expand_as(noisy_masked)] = 0.0   # replace with 0 (could also use neighbor fill)
    return noisy_masked, mask

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred
