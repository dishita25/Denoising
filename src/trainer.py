import torch.optim as optim
from src.loss import loss_func
#from src.model.ZSN2N import network
from src.model.BSZSN2N import network
from src.dataloader.PolyU import evaluate_polyu
from src.dataloader.CBSD68 import evaluate_artificial
from src.dataloader.noisy_dataset import NoisyImageDataset
from src.utils import test
import torchvision.transforms as T 
from torch.utils.data import DataLoader
from src.dataloader.speech_dataset import SpeechDataset
from src.dataloader.speech_dataset import train_input_files, train_target_files, test_noisy_files, test_clean_files
import tqdm
from pesq import pesq
from scipy import interpolate
import torch
import numpy as np
from src.device import DEVICE
from metrics import AudioMetrics2


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
    batch_size,
    device="cuda",
):
    model = network(n_chan).to(device)
    print("The number of parameters of the network is:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    transform = T.Compose([
        T.CenterCrop((256,256)),
        T.ToTensor()
    ])

    # train_dataset = NoisyImageDataset(noisy_dir=noisy_dir, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    train_dataset = SpeechDataset(train_input_files, train_target_files)
    test_dataset = SpeechDataset(test_noisy_files, test_clean_files)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4) 



    # for epoch in range(1, max_epoch + 1):
    #     model.train()
    #     epoch_loss = 0.0
        
    #     for batch_idx, noisy_imgs in enumerate(train_loader):
    #         noisy_imgs = noisy_imgs.to(device)
            
    #         batch_loss = 0.0
    #         for i in range(noisy_imgs.size(0)):
    #             noisy_img = noisy_imgs[i:i+1]
    #             loss = loss_func(noisy_img, model, mask_ratio=mask_ratio, blind_spot_weight=blind_spot_weight)
    #             batch_loss += loss
            
    #         batch_loss = batch_loss / noisy_imgs.size(0)
            
    #         optimizer.zero_grad()
    #         batch_loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += batch_loss.item()

    #     scheduler.step()
    #     epoch_loss /= len(train_loader)

    #     if epoch % 100 == 0 or epoch == 1:
    #         print(f"Epoch [{epoch}/{max_epoch}] | Loss: {epoch_loss:.6f}")

    # return model
    for epoch in range(1, max_epoch + 1):
        model.train()
        epoch_loss = 0.0
        epoch_res_loss = 0.0
        epoch_cons_loss = 0.0
        
        # Use tqdm for better progress tracking
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epoch}', leave=False)
        
        for batch_idx, (noisy_stft, clean_stft) in enumerate(pbar):
            noisy_stft = noisy_stft.to(device)
            # Note: clean_stft not used in ZSN2N training
            
            batch_loss = 0.0
            batch_res_loss = 0.0
            batch_cons_loss = 0.0
            
            # Process each STFT sample individually
            for i in range(noisy_stft.size(0)):
                single_noisy_stft = noisy_stft[i:i+1]
                
                # Calculate ZSN2N loss
                if isinstance(loss_func(single_noisy_stft, model, mask_ratio, blind_spot_weight), tuple):
                    total_loss, res_loss, cons_loss = loss_func(single_noisy_stft, model, mask_ratio, blind_spot_weight)
                    batch_res_loss += res_loss.item()
                    batch_cons_loss += cons_loss.item()
                else:
                    total_loss = loss_func(single_noisy_stft, model, mask_ratio, blind_spot_weight)
                
                batch_loss += total_loss
            
            # Average and backpropagate
            batch_loss = batch_loss / noisy_stft.size(0)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        return model


# Repeating at a lot of places
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    # Convert to complex before istft
    y_true_complex = torch.complex(y_true_[..., 0], y_true_[..., 1])
    y_true = torch.istft(y_true_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    
    x_ = torch.squeeze(x_, 1)
    # Convert to complex before istft
    x_complex = torch.complex(x_[..., 0], x_[..., 1])
    x = torch.istft(x_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","
        #print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            if use_net: # Forward of net returns the istft version
                x_est = net(noisy.to(DEVICE), is_istft=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                noisy_squeezed = torch.squeeze(noisy, 1)
                noisy_complex = torch.complex(noisy_squeezed[..., 0], noisy_squeezed[..., 1])
                x_est_np = torch.istft(noisy_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

            clean_squeezed = torch.squeeze(clean, 1)
            clean_complex = torch.complex(clean_squeezed[..., 0], clean_squeezed[..., 1])
            x_clean_np = torch.istft(clean_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

        
            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)
            
            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
            
            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            #print(new_scores)
            #print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results



def test_model(model, dataset_name, dataset_path, device="cuda", noise_level = None):
    # model.eval()

    # if dataset_name == "polyu":
    #     print("Evaluating on PolyU...")
    #     psnr, ssim = evaluate_polyu(model, dataset_path, device)

    # elif dataset_name == "Mcmaster" or dataset_name == "CBSD" or dataset_name == "kodak":
    #     print("Evaluating on Artificial dataset...")
    #     psnr, ssim = evaluate_artificial(model, dataset_name, noise_level, dataset_path, device)

    # else:
    #     raise ValueError(f"Unknown dataset: {dataset_name}")

    # return psnr, ssim
    model.eval()
    test_ep_loss = 0.
    counter = 0.
    '''
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter
    '''
    
    #print("Actual compute done...testing now")
    
    testmet = getMetricsonLoader(test_loader,net,use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, testmet


