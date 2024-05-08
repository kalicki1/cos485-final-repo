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
    return [wav]

def main(wiener_window, source_folder, target_folder):

    for filename in tqdm(os.listdir(source_folder), desc='Wiener filter over source audio files'):
        # file = filename.split('.')[0]
        file = filename
        file_path = source_folder + file

        noisy_slices = slice_signal(file_path, window_size, 1, sample_rate)

        # Run Wiener filter on sliced signal
        enhanced_speech = wiener(noisy_slices, wiener_window)

        # Run Wiener filter on every slice in sliced siganl
        enhanced_speech = wiener(noisy_slices, wiener_window)
        enhanced_speech = []
        # for noisy_slice in tqdm(noisy_slices, desc='Wiener filter over audio files'):
        for noisy_slice in noisy_slices:
            noisy_slice = noisy_slice[np.newaxis, np.newaxis, :]
            filtered_speech = wiener(noisy_slice, wiener_window) 
            filtered_speech = filtered_speech.reshape(-1)
            enhanced_speech.append(filtered_speech)

        enhanced_speech = np.array(enhanced_speech).reshape(1, -1)

        # Save Wiener-filtered audio
        filtered_path = target_folder + file
        # filtered_file = os.path.join(os.path.dirname(file),
                                # 'wiener_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
        wavfile.write(filtered_path, sample_rate, enhanced_speech.T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('-f', '--filter-window', type=int, default=2, help='Wiener filter window')
    parser.add_argument('-s', '--source', type=str, required=True, help='Source folder for audio to stitch')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target folder for stitched audio')

    opt = parser.parse_args()
    FILTER_WINDOW = opt.filter_window
    SOURCE = opt.source
    TARGET = opt.target

    # MACOS only
    if SOURCE[-1] != '/':
        SOURCE += '/'
    if TARGET[-1] != '/':
        TARGET += '/'

    if not os.path.exists(SOURCE):
        os.makedirs(SOURCE)
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    main(FILTER_WINDOW, SOURCE, TARGET)