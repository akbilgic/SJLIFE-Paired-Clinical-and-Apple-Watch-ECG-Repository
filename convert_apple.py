import numpy as np
from scipy import signal

apple_ecg = np.load('AppleECGs_full_243/apple_ecg_132.npy')

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
output_path = './500hz_apple/500hz_apple_ecg_132.npy'
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
output_segment_path = './500hz_10sec_apple/500hz_10sec_apple_ecg_132.npy'
np.save(output_segment_path, ecg_segment)
print(f"Segmented ECG saved to: {output_segment_path}")