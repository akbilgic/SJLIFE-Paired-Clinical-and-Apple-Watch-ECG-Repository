import numpy as np

apple_ecg = np.load('AppleECGs_full_243/apple_ecg_132.npy')
print("Apple ECG Shape:")
print(apple_ecg.shape)
print("Apple ECG:")
print(apple_ecg)
print("######")

clinical_ecg = np.load('ClinicalECGs_full_243/clinical_ecg_132.npy')
print("Clinical ECG Shape:")
print(clinical_ecg.shape)
print("Clinical ECG:")
print(clinical_ecg)
######
