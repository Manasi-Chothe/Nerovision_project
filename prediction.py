import numpy as np
import tensorflow as tf
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = tf.keras.models.load_model('eeg_cnn_model.h5')
model.summary()

# Load and preprocess new EEG data (replace with the actual path)
new_file_path = r'name0_2.edf'  # Replace with actual path
new_raw_data = mne.io.read_raw_edf(new_file_path, preload=True)

# Apply the same filter and preprocessing steps as during training
new_filtered_data = new_raw_data.filter(l_freq=1.0, h_freq=50.0)

# Segment the new data into 2-second epochs
new_events = mne.make_fixed_length_events(new_filtered_data, duration=2.0)
new_epochs = mne.Epochs(new_filtered_data, new_events, tmin=0, tmax=2, baseline=None, preload=True)

# Load training data to fit the scaler (ensure the training data is available)
train_data = np.load('X_preprocessed.npy')  # Replace with the correct path to your training data

# Reshape the training data for fitting the scaler
train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])

# Create and fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(train_data_reshaped)

# Optionally save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use (if desired)

# Normalize the new EEG data using the fitted scaler
new_eeg_data = new_epochs.get_data()
new_eeg_data_normalized = scaler.transform(new_eeg_data.reshape(-1, new_eeg_data.shape[-1])).reshape(new_eeg_data.shape)

# Reshape the data for CNN input (adding channel dimension)
new_eeg_data_reshaped = new_eeg_data_normalized[..., np.newaxis]  # Adding channel dimension

# Make predictions on the new EEG data
predictions = model.predict(new_eeg_data_reshaped)

# Output predictions (e.g., 0 or 1 for binary classification)
print(predictions)

# Convert predictions to class labels (if binary classification)
predicted_labels = (predictions > 0.5).astype(int)

# Print the predicted labels
print("Predicted labels:", predicted_labels)

# Plot the first epoch's EEG signal with its predicted label
epoch_idx = 0  # You can choose any epoch index to visualize
plt.figure(figsize=(10, 5))
plt.plot(new_eeg_data_reshaped[epoch_idx, 0, :])  # Plot the first channel of the selected epoch
plt.title(f'Predicted Digit: {predicted_labels[epoch_idx][0]}')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Save predictions for later use
np.save('predictions.npy', predictions)
np.save('predicted_labels.npy', predicted_labels)
print("Predictions saved!")
