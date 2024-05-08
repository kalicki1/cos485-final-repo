import os
import argparse

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# stitch_folder = '../results/noisy_baseline/'
# target_folder = '../results/noisy_wiener/'
window_size = 2 ** 14  # about 1 second of samples
sample_rate = 16000
stride = 0.5


def stitch_audios_in_folder_over_epoch(stitch_folder, target_folder):
    """
    Stitch the chopped up audio in each folder, according to stride, back into one
    """
    epochs = {}    # To store the sorted audio files

    highest_epoch = 0
    # Organize into bins according to target file
    for filename in os.listdir(stitch_folder):
        # sample_name = filename.split('.')[0]
        einstance = filename.split('e')[1]
        einstance = einstance.split('.')[0]
        einstance = int(einstance)

        if einstance > highest_epoch: 
            highest_epoch = einstance
        if einstance not in epochs:
            epochs[einstance] = []
        epochs[einstance].append(filename)

    # organize each epoch instance according their audio type
    for epoch_key in epochs:
        audio_files = {}
        for filename in epochs[epoch_key]:
            sample_name = filename.split('.')[0]
            if sample_name not in audio_files:  # Create the list if its not see
                audio_files[sample_name] = []
            audio_files[sample_name].append(filename)   # Append
        epochs[epoch_key] = audio_files
            
    # create target folders for each epoch
    for i in range(1, highest_epoch+1):
        i = str(i)
        epoch_dir = os.path.join(TARGET, i)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)


    # Sort dict bins alphanumerically 
    for epoch_key in epochs:
        for audio_sample in epochs[epoch_key]:
            epochs[epoch_key][audio_sample].sort()

    # Stitch together and generate wav for each target
    overlap_samples = int(window_size * stride)     # also the 'hop'

    # For every epoch
    for epoch_key in tqdm(epochs, desc='Stitch all samples in each epoch'):

        # Stitch together all the samples
        audio_files = epochs[epoch_key]
        for audio_sample in tqdm(audio_files, desc='stitch together components'):
            if audio_sample == '':
                continue
            components = audio_files[audio_sample]
            combined_data = None  # Store the output of the stitched file

            # for each component, 
            for idx, wav_file in enumerate(components):
                # Load the WAV file
                wav_path = stitch_folder + wav_file
                if wav_path.split('.')[-1] != 'wav':
                    continue
                data, sr = librosa.load(wav_path, sr=None)

                # Remove overlap from all files except the first one
                if idx != 0:
                    data = data[overlap_samples:]
        
                # Concatenate data
                if combined_data is None:
                    combined_data = data
                else:
                    combined_data = np.concatenate((combined_data, data))

            # Write the combined data to a new WAV file
            target_file = '{}{}.wav'.format(target_folder, audio_sample)
            # print("targetfile: ", target_file)
            sf.write(target_file, combined_data, sr)


def stitch_audios_in_folder(stitch_folder, target_folder):
    """
    Stitch the chopped up audio in each folder, according to stride, back into one
    """
    audio_files = {}    # To store the sorted audio files

    # Organize into bins according to target file
    for filename in os.listdir(stitch_folder):
        sample_name = filename.split('.')[0]
        if sample_name not in audio_files:  # Create the list if its not see
            audio_files[sample_name] = []
        audio_files[sample_name].append(filename)   # Append

    # Sort dict bins alphanumerically 
    for audio_sample in audio_files:
        audio_files[audio_sample].sort()

    # Stitch together and generate wav for each target
    overlap_samples = int(window_size * stride)     # also the 'hop'

    for audio_sample in tqdm(audio_files):
        if audio_sample == '':
            continue
        components = audio_files[audio_sample]
        combined_data = None  # Store the output of the stitched file

        # for each component, 
        for idx, wav_file in enumerate(components):
            # Load the WAV file
            wav_path = stitch_folder + wav_file
            if wav_path.split('.')[-1] != 'wav':
                continue
            data, sr = librosa.load(wav_path, sr=None)

            # Remove overlap from all files except the first one
            if idx != 0:
                data = data[overlap_samples:]
        
            # Concatenate data
            if combined_data is None:
                combined_data = data
            else:
                combined_data = np.concatenate((combined_data, data))

        # Write the combined data to a new WAV file
        target_file = '{}{}.wav'.format(target_folder, audio_sample)
        # print("targetfile: ", target_file)
        sf.write(target_file, combined_data, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('-s', '--source', type=str, required=True, help='Source folder for audio to stitch')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target folder for stitched audio')

    opt = parser.parse_args()
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

    stitch_audios_in_folder_over_epoch(SOURCE, TARGET)
