# # METHODS OF EVALUATION
# 
# STOI
# PESQ
# SNR
import argparse
import os
from tqdm import tqdm

from scipy.io import wavfile
from pesq import pesq


################################################################################

def main(clean_folder, test_folder, log_file):

    # Set up log file
    with open(log_file, 'w') as f:
        f.write("FILE\tPESQ\n")

    # Evaluate metrics for each test file
    for file in tqdm(os.listdir(test_folder)): 
        test_path = os.path.join(test_folder, file)
        clean_path = os.path.join(clean_folder, file)
        if not os.path.exists(clean_path):
            print("Warning! {} does not have a clean analogue!".format(file))
            continue

        # Note: clean => reference and degraded => test
        # Compute PESQ score
        rate, ref = wavfile.read(clean_path)
        rate, deg = wavfile.read(test_path)
        pesq_score = pesq(rate, ref, deg, 'wb')

        # Append all metrics
        with open(log_file, 'a') as f:
            f.write('%s\t%s\n' % 
                    (file, 
                    pesq_score))
        
    print("Log file: {}".format(os.path.abspath(log_file)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('-c', '--clean_folder', type=str, required=True, help='audio file name')
    parser.add_argument('-t', '--test_folder', type=str, required=True, help='generator epoch name')
    parser.add_argument('-l', '--log-file', type=str, required=True, help="where to save output")

    opt = parser.parse_args()
    clean_folder = opt.clean_folder
    test_folder = opt.test_folder
    log_file = opt.log_file

    print("Startup note: Every test file should have a corresponding clean file (same name).")

    main(clean_folder, test_folder, log_file)