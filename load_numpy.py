import numpy as np
import matplotlib.pyplot as plt

# APPLE ECG
apple_ecg = np.load('./500hz_10sec_apple/500hz_10sec_apple_ecg_132.npy')
print("Apple ECG Shape:")
print(apple_ecg.shape)
print("Apple ECG:")
print(apple_ecg)
print("######")


plt.plot(apple_ecg.flatten())
plt.title("Apple ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# CLINICAL ECG
clinical_ecg = np.load('first_lead_clinical/first_lead_clinical_ecg_132.npy')
print("Clinical ECG Shape:")
print(clinical_ecg.shape)
print("Clinical ECG:")
print(clinical_ecg)

plt.plot(clinical_ecg.flatten())
plt.title("Clinical ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
