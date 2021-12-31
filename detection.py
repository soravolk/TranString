import torchaudio
import pyaudio
from pyaudio import paContinue, paInt32
import wave
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

CHUNK = 44100 # window size
RATE = 44100 # smaple rate
RECORD_SECONDS = 60
FORMAT = paInt32
filename = "output.wav"
config = {
    # first 3 keys are dataset dependent
    "identifier": "electric guitar",
    # hop length for the FFT (number of samples between successive frames)
    "hop_length": 441,
    # number of frequency bins of the FFT (frame size = length of FFT window)
    "n_fft": 2048,
    # number of mel bands  
    "n_mels": 128,
    # "resample_sr": 16000
}
label_convert= {0 : 'bending', 
         1 : 'trill', 
         2 : 'sliding',
         3 : 'pulling',
         4 : 'normal',
         5 : 'mute',
         6 : 'hammering',}

# set model
print('Load model ...')
model = models.vgg11_bn(pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1))
model.classifier[6] = nn.Linear(4096, 7)

path = f"state_dict/VGG11_bn_pretrain/best_acc_model_0.947.ckpt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(path, map_location=device))
print('Finish')

def mel_spectrogram(audio_np, down_rate):
    downsample = torchaudio.transforms.Resample(
        orig_freq=RATE, new_freq=RATE//down_rate)
    audio = torch.from_numpy(audio_np.astype(np.float32))
    #print(audio)
    audio = downsample(audio)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = RATE//down_rate,
        hop_length = config["hop_length"]//down_rate,
        n_fft = config["n_fft"]//down_rate,
        n_mels = config["n_mels"])(audio).T
    mel_spectrogram = np.log10(1 + 10 * mel_spectrogram)

    return mel_spectrogram.T

def pred_technique(feature):
    model.eval()
    test_pred = 'X'
    with torch.no_grad():
        X = feature.unsqueeze(0).unsqueeze(0).to(device)
        output = model(X)
        _, test_pred = torch.max(output, 1)
        
        print(label_convert[test_pred.numpy()[0]])

def rec(file_name):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("Start Recording......")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        data_array = np.frombuffer(data, dtype=np.int32)
        #data_array = data_array/np.linalg.norm(data_array)
        a = data_array - data_array.mean()
        data_array = a/np.abs(a).max()
        mel_spec = mel_spectrogram(data_array, down_rate=2)
        pred_technique(mel_spec)
        #stream.write(data)
    print("Recording ends......")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

rec('output.wav')