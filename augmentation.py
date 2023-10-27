import torchaudio
import torch
import random
import numpy as np 


def speed_perturbation(audio, sample_rate, speed_factor_range=(0.9, 1.1)):
    speed_factor = random.uniform(speed_factor_range[0], speed_factor_range[1])
    
    resampled_audio = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=int(sample_rate * speed_factor)
    )(audio)
    
    return resampled_audio, int(sample_rate * speed_factor)


def time_shifting(audio, sample_rate, shift_factor_range=(-0.5, 0.5)):
    shift_factor = random.uniform(shift_factor_range[0], shift_factor_range[1])
    
    shift_samples = int(sample_rate * shift_factor)
    shifted_audio = torch.roll(audio, shifts=shift_samples, dims=-1)
    
    return shifted_audio


def additive_noise(audio, sample_rate, noise_factor=0.1):
    noise = torch.randn_like(audio) * noise_factor
    noisy_audio = audio + noise
    
    return noisy_audio


def pitch_perturbation(audio, sample_rate, pitch_factor_range=(-0.2, 0.2)):
    pitch_factor = random.uniform(pitch_factor_range[0], pitch_factor_range[1])
    
    pitch_shifted_audio = torchaudio.transforms.PitchShift(
        sample_rate=sample_rate,
        n_steps=pitch_factor
    )(audio)
    
    return pitch_shifted_audio


def volume_perturbation(audio, sample_rate, volume_factor_range=(0.8, 1.2)):
    volume_factor = random.uniform(volume_factor_range[0], volume_factor_range[1])
    scaled_audio = audio * volume_factor
    
    return scaled_audio


def reverberation(audio, sample_rate, room_decay_factor=0.5):
    reverberant_audio = torchaudio.transforms.Reverb(
        sample_rate=sample_rate,
        room_decay=room_decay_factor
    )(audio)
    
    return reverberant_audio


def time_stretching(audio, sample_rate, stretch_factor_range=(0.8, 1.2)):
    stretch_factor = random.uniform(stretch_factor_range[0], stretch_factor_range[1])
    
    stretched_audio = torchaudio.transforms.TimeStretch(
        fixed_rate=stretch_factor
    )(audio)
    
    return stretched_audio


def quick_speed_perturbation(audio, sample_rate):
    speed_factors = [0.9, 1.1, 1.2, 0.8]  
    speed_factor = random.choice(speed_factors)
    
    resampled_audio = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=int(sample_rate * speed_factor)
    )(audio)
    
    return resampled_audio, int(sample_rate * speed_factor)


def in_fading_out_fading(audio, sample_rate, fade_in_duration=1.0, fade_out_duration=1.0):
    fade_in_samples = int(sample_rate * fade_in_duration)
    fade_out_samples = int(sample_rate * fade_out_duration)
    
    in_fading_audio = audio.clone()
    in_fading_audio[..., :fade_in_samples] *= torch.linspace(0, 1, fade_in_samples)
    
    out_fading_audio = audio.clone()
    out_fading_audio[..., -fade_out_samples:] *= torch.linspace(1, 0, fade_out_samples)
    
    return in_fading_audio, out_fading_audio


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


augmentations = [
    speed_perturbation, 
    time_shifting, 
    additive_noise, 
    in_fading_out_fading,
    quick_speed_perturbation,
    time_stretching,
    reverberation,
    volume_perturbation,
    pitch_perturbation
]



if __name__ == "__main__":
    file_path = 'path_to_your_audio_file.wav'
    waveform, sample_rate = torchaudio.load(file_path)

    augmented_waveform = pitch_perturbation(waveform, sample_rate)

    augmented_file_path = 'path_to_save_augmented_audio_file.wav'
    print(sample_level(5))
    