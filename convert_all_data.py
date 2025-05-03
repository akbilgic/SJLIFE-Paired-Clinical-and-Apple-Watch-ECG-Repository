import numpy as np
from scipy import signal
import glob
import os

def apple_convert(apple_ecg, apple_ecg_name):
    # Extract filename from full path
    filename = os.path.basename(apple_ecg_name)

    # parameters
    orig_fs = 512     # original sampling rate, Hz
    target_fs = 500   # desired sampling rate, Hz
    duration = 30     # seconds

    # compute number of samples
    orig_n = orig_fs * duration      # 15360
    target_n = target_fs * duration  # 15000

    # FFT-based resampling
    ecg_resampled = signal.resample(apple_ecg, target_n)
    print("Resampled (SciPy) shape:", ecg_resampled.shape)
    
    # Save the resampled ECG as a numpy file
    output_path = f'./500hz_apple/500hz_{filename}'
    np.save(output_path, ecg_resampled)
    print(f"Resampled ECG saved to: {output_path}")

    # Select 5000 - 10000 samples means 10 seconds
    start_sample = 5000
    end_sample = 10000
    ecg_segment = ecg_resampled[start_sample:end_sample]
    print("Segmented ECG shape:", ecg_segment.shape)
    print("Segmented ECG:", ecg_segment)

    # Reshape to (1, 1, L)
    ecg_segment_reshaped = ecg_segment.reshape(1, 1, -1)
    print("Reshaped ECG shape:", ecg_segment_reshaped.shape)
    print("Reshaped ECG:", ecg_segment_reshaped)

    # Save the reshaped ECG as a numpy file
    output_segment_path = f'./500hz_10sec_apple/500hz_10sec_{filename}'
    np.save(output_segment_path, ecg_segment_reshaped)
    print(f"Segmented ECG saved to: {output_segment_path}")


def clinical_convert(clinical_ecg, clinical_ecg_name):
    # Extract filename from full path
    filename = os.path.basename(clinical_ecg_name)

    first_lead = clinical_ecg[:, 0:1, :] # select the first lead
    print("First Lead ECG Shape:")
    print(first_lead.shape)
    print("First Lead ECG:")
    print(first_lead)

    # Save the first lead ECG as a numpy file
    output_path = f'./first_lead_clinical/first_lead_{filename}'
    np.save(output_path, first_lead)
    print(f"First lead ECG saved to: {output_path}")


def main_loop(apple_ecg_dir, clinical_ecg_dir):
    apple_files = sorted(glob.glob(os.path.join(apple_ecg_dir, 'apple_ecg_*.npy')))
    clinical_files = sorted(glob.glob(os.path.join(clinical_ecg_dir, 'clinical_ecg_*.npy')))
    
    for apple_file, clinical_file in zip(apple_files, clinical_files):
        print(f"Processing {apple_file} and {clinical_file}")
        apple_ecg = np.load(apple_file)
        clinical_ecg = np.load(clinical_file)
        
        apple_convert(apple_ecg, apple_file)
        clinical_convert(clinical_ecg, clinical_file)
        print("######")

if __name__ == "__main__":
    # Define the directories for Apple and Clinical ECG data
    apple_ecg_dir = './AppleECGs_full_243'
    clinical_ecg_dir = './ClinicalECGs_full_243'
    
    # Call the main loop
    main_loop(apple_ecg_dir, clinical_ecg_dir)

