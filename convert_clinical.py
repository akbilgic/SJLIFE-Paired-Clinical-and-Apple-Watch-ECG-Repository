import numpy as np

clinical_ecg = np.load('ClinicalECGs_full_243/clinical_ecg_132.npy')
print("Clinical ECG Shape:")
print(clinical_ecg.shape)
print("Clinical ECG:")
print(clinical_ecg)
print("######")


first_lead = clinical_ecg[:, 0:1, :] # select the first lead
print("First Lead ECG Shape:")
print(first_lead.shape)
print("First Lead ECG:")
print(first_lead)

# Save the first lead ECG as a numpy file
output_path = './first_lead_clinical/first_lead_clinical_ecg_132.npy'
np.save(output_path, first_lead)
print(f"First lead ECG saved to: {output_path}")
