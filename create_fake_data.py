import numpy as np

samples = 500
time_steps = 128
channels = 32
classes = 3

X = np.random.randn(samples, time_steps, channels)
y = np.random.randint(0, classes, size=(samples,))

np.save("data/demo_eeg.npy", {"X": X, "y": y})
print("Fake EEG dataset saved to data/demo_eeg.npy")