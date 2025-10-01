from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

noise_class = "white" 
training_type =  "Noise2Noise" 

import os
basepath = str(noise_class)+"_"+training_type
os.makedirs(basepath,exist_ok=True)
os.makedirs(basepath+"/Weights",exist_ok=True)
os.makedirs(basepath+"/Samples",exist_ok=True)

if noise_class == "white": 
    TRAIN_INPUT_DIR = Path('/kaggle/working/Audio_Denoising/Datasets/WhiteNoise_Train_Input')

    if training_type == "Noise2Noise":
        TRAIN_TARGET_DIR = Path('/kaggle/working/Audio_Denoising/Datasets/WhiteNoise_Train_Output')
    elif training_type == "Noise2Clean":
        TRAIN_TARGET_DIR = Path('/kaggle/working/Audio_Denoising/Datasets/clean_trainset_28spk_wav')
    else:
        raise Exception("Enter valid training type")

    TEST_NOISY_DIR = Path('/kaggle/working/Audio_Denoising/Datasets/WhiteNoise_Test_Input')
    TEST_CLEAN_DIR = Path('/kaggle/working/Audio_Denoising/Datasets/clean_testset_wav') 
    
else:
    TRAIN_INPUT_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Train_Input')

    if training_type == "Noise2Noise":
        TRAIN_TARGET_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Train_Output')
    elif training_type == "Noise2Clean":
        TRAIN_TARGET_DIR = Path('Datasets/clean_trainset_28spk_wav')
    else:
        raise Exception("Enter valid training type")

    TEST_NOISY_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Test_Input')
    TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav') 

    
class SpeechDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.len_ = len(self.noisy_files)
        
        # fixed len
        self.max_len = 165000

    
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform
  
    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        # Short-time Fourier transform
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, 
                                hop_length=self.hop_length, normalized=True,
                                return_complex=False)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, 
                                hop_length=self.hop_length, normalized=True,
                                return_complex=False)

        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        
        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)
        
        return output
    
train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

print("No. of Training files:",len(train_input_files))
print("No. of Testing files:",len(test_noisy_files))