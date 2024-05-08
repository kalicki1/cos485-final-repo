import argparse
import os

import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import wiener
from tqdm import tqdm

window_size = 2 ** 14  # about 1 second of samples
sample_rate = 16000

def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices

if __name__ == '__main__':
    # CLEAN_PATH = '../datasets/DS_10283_2791_subset/clean_testset_wav/'
    NOISY_PATH = '../datasets/DS_10283_2791_subset/noisy_testset_wav/'
    DOWNSAMPLE_NOISY_PATH = '../datasets/DS_10283_2791_subset/noisy_test_downsample_wav/'

    FILE_NAME = 'p232_001.wav'
    NOISY_FILE = NOISY_PATH + FILE_NAME

    noisy_slices = slice_signal(NOISY_FILE, window_size, 1, sample_rate)

    # Run Wiener filter on sliced signal
    print("Running wiener filter on {}.".format(NOISY_FILE))
    enhanced_speech = wiener(noisy_slices, WIENER_FILTER_WINDOW)

    # Run Wiener filter on every slice in sliced siganl
    enhanced_speech = wiener(noisy_slices, WIENER_FILTER_WINDOW)
    enhanced_speech = []
    count = 0
    for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
        print("count: {}".format(count))
        noisy_slice = noisy_slice[np.newaxis, np.newaxis, :]
        filtered_speech = wiener(noisy_slice, WIENER_FILTER_WINDOW) 
        filtered_speech = filtered_speech.reshape(-1)
        enhanced_speech.append(filtered_speech)
        count += 1

    enhanced_speech = np.array(enhanced_speech).reshape(1, -1)

    # Save Wiener-filtered audio
    CLEAN_FILE = CLEAN_PATH + FILE_NAME
    file_name = os.path.join(os.path.dirname(CLEAN_FILE),
                             'wiener_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, enhanced_speech.T)
    print("Saved Wiener-filtered file to {}".format(CLEAN_FILE))