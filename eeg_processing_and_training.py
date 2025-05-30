import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the .edf file
file_path = r'name0_2.edf'  # Replace with the path to your .edf file
raw_data = mne.io.read_raw_edf(file_path, preload=True)

# Print some basic info about the data
print(raw_data.info)

# Step 2: Apply bandpass filter (1 to 50 Hz)
low_cutoff = 1.0
high_cutoff = 50.0
filtered_data = raw_data.filter(l_freq=low_cutoff, h_freq=high_cutoff)

# Plot the filtered data to visualize the results
filtered_data.plot(duration=5, n_channels=30)
plt.show()

# Step 3: Segment the data into 2-second epochs
events = mne.make_fixed_length_events(filtered_data, duration=2.0)
epochs = mne.Epochs(filtered_data, events, tmin=0, tmax=2, baseline=None, preload=True)

# Plot a few epochs to visualize the segmentation
epochs.plot(n_epochs=5, n_channels=30)
plt.show()

# Step 4: Normalize the EEG data
# Get the segmented EEG data
eeg_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

# Normalize each channel using StandardScaler
scaler = StandardScaler()
eeg_data_normalized = scaler.fit_transform(eeg_data.reshape(-1, eeg_data.shape[-1])).reshape(eeg_data.shape)

# Replace the original data with the normalized data
epochs._data = eeg_data_normalized

# Step 5: Visualize the normalized data (optional)
for epoch in range(5):  # Plot a few epochs
    plt.figure(figsize=(10, 5))
    plt.plot(eeg_data_normalized[epoch, 0, :])  # Plotting the first channel as an example
    plt.title(f'Epoch {epoch + 1} - First Channel')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Step 6: Prepare Data for Machine Learning (CNN or other models)
# X is the input data (EEG signals)
X = epochs.get_data()  # This will give you the shape (n_epochs, n_channels, n_times)

# Placeholder labels (y) - you should replace this with your actual labels
# For now, this just creates dummy labels, adjust based on your experiment
y = np.random.randint(0, 2, size=(X.shape[0],))  # Random binary labels as a placeholder

# Print shape of the data
print(f"EEG data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# After preprocessing, save X and y for CNN
np.save('X_preprocessed.npy', X)  # Save preprocessed EEG data
np.save('y_labels.npy', y)         # Save corresponding labels

print("Preprocessed data and labels saved!")
