import numpy as np
# import librosa #has to do this cause librosa is not supported on my server
import librosa
import torch
import torchaudio as ta
import torchaudio.functional as ta_F
import torchaudio.transforms as ta_T
from WavLM.WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
import soundfile as sf
# import pyloudnorm as pyln


def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = chunk_size // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio


def  get_mfcc_ta(audio_fn, eps=1e-6, fps=15, smlpx=False, sr=16000, n_mfcc=64, win_size=None, type='mfcc', am=None, am_sr=None, encoder_choice='mfcc'):

    speech_array, sampling_rate = librosa.load(audio_fn, sr=16000)
  
    if encoder_choice == 'faceformer':
            audio_ft = speech_array.reshape(-1, 1)
    elif encoder_choice == 'meshtalk':
            audio_ft = 0.01 * speech_array / np.mean(np.abs(speech_array))
    elif encoder_choice == 'onset':
            audio_ft = librosa.onset.onset_detect(y=speech_array, sr=16000, units='time').reshape(-1, 1)
    else:
           
            audio, sr_0 = ta.load(audio_fn)
            if sr != sr_0:
                audio = ta.transforms.Resample(sr_0, sr)(audio)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            n_fft = 2048
           
            if fps == 15:
                hop_length = 1070
            elif fps == 30:
                hop_length = 535  ###16000/30(fps)
            win_length = hop_length * 2
            n_mels = 256
            n_mfcc = 64

            mfcc_transform = ta_T.MFCC(
                sample_rate=sr,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    # "win_length": win_length,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )
           
            audio_ft = mfcc_transform(audio).squeeze(dim=0).transpose(0, 1).numpy()
            audio_align_id = librosa.onset.onset_detect(y=speech_array, sr=16000, units='time').reshape(-1)
            audio_align_id = np.trunc(np.dot(30,audio_align_id)).astype(int)
            filtered_audio_align_id = audio_align_id[audio_align_id <= audio_ft.shape[0]-1]
            #print('filtered_audio_align_id:',filtered_audio_align_id.shape,audio_align_id.shape)
            audio_align = np.zeros((audio_ft.shape[0],1))
            audio_align[filtered_audio_align_id] = 1
            audio_ft = np.concatenate([audio_align,audio_ft],axis=1)
 
    return audio_ft


def wavlm_init():
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, len):
    with torch.no_grad():
        #wav_input_16khz = torch.from_numpy(wav_input_16khz).float()
        wav_input_16khz = wav_input_16khz.unsqueeze(0)
        rep = model.extract_features(wav_input_16khz)[0]
        rep = F.interpolate(rep.transpose(1, 2), size=len, align_corners=True, mode='linear').transpose(1, 2)
        return rep.detach().data
    
def get_wavlm(model,wav,len):
    #wav, fs = sf.read(audio_fn)
    sample_wavlm = wav2wavlm(model,wav,len)
    return sample_wavlm
