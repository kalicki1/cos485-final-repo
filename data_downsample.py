import os

import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

clean_test_folder = '../datasets/DS_10283_2791_subset/clean_testset_wav'
noisy_test_folder = '../datasets/DS_10283_2791_subset/noisy_testset_wav'
serialized_noisy_train_folder = '../datasets/DS_10283_2791_subset/serialized_noisy_train_data'
serialized_noisy_test_folder = '../datasets/DS_10283_2791_subset/serialized_noisy_test_data'
serialized_clean_train_folder = '../datasets/DS_10283_2791_subset/serialized_clean_train_data'
serialized_clean_test_folder = '../datasets/DS_10283_2791_subset/serialized_clean_test_data'
target_noisy_folder = '../results/baseline_noisy_sliced'
target_clean_folder = '../results/baseline_clean_sliced'

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


def process_and_serialize_and_to_wav(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5

    clean_folder = clean_test_folder
    noisy_folder = noisy_test_folder
    serialized_noisy_folder = serialized_noisy_test_folder
    serialized_clean_folder = serialized_noisy_test_folder

    if not os.path.exists(serialized_noisy_folder):
        os.makedirs(serialized_noisy_folder)

    # walk through the path, slice the audio file, and save the serialized result
    for root, dirs, files in os.walk(noisy_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            # for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
            for idx, slice_tuple in enumerate(noisy_sliced):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_noisy_folder, '{}_{}'.format(filename, idx)), arr=pair)

                noisy_sliced = np.array(noisy_sliced).reshape(1, -1)
                file_name = os.path.join(target_noisy_folder,
                                        '{}_{}.wav'.format(filename, idx))
                wavfile.write(file_name, sample_rate, slice_tuple.T)
            for idx, slice_tuple in enumerate(clean_sliced):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(serialized_clean_folder, '{}_{}'.format(filename, idx)), arr=pair)

                clean_sliced = np.array(clean_sliced).reshape(1, -1)
                file_name = os.path.join(target_clean_folder,
                                        '{}_{}.wav'.format(filename, idx))
                wavfile.write(file_name, sample_rate, slice_tuple.T)

if __name__ == '__main__':

    if not os.path.exists(target_noisy_folder):
        os.makedirs(target_noisy_folder)
    if not os.path.exists(target_clean_folder):
        os.makedirs(target_clean_folder)

    process_and_serialize_and_to_wav('test')
