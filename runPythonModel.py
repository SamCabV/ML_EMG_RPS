import numpy as np
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.signal import welch

import joblib

# Load the model from file


class RunPythonModel:
    def __init__(self, modelPath):
        self.model = None # replace this with whatever code you need to load model
        # Example usage
        self.included_features =  ['WL', 'RMS', 'SSI', 'MeanPower', 'MedianFreq']#, 'WL', 'RMS', 'SSI', 'var', 'MedianFreq']
        self.fs = 1000  # Sampling frequency
        self.model_file_name = 'my_random_forest_model.joblib'
        self.model = joblib.load(self.model_file_name)
        self.tags = {0: 'rock', 1: 'paper', 2: 'scissors', 3: 'Resting'}
    
    def get_rps(self,data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """

        transformed_data = transform_data(data).T
        filtered_data = filter_data(transformed_data, fs)
        #print(np.shape(filtered_data))

        feature_table = extract_features(filtered_data, self.included_features, self.fs)
        print(feature_table)
        print(np.shape(feature_table))
        data_list = []

    # Iterate over the dictionary and append the data to the list
        for key in feature_table:
            data_list.append(feature_table[key])
        FT_Final =  np.array(data_list).T
        print(np.shape(FT_Final))

        # Convert the list of arrays into a NumPy array
        prediction = self.model.predict(FT_Final)
        return self.tags[int(prediction[0])]
 

def transform_data(lsl_data):
    # Assuming lsl_data is a numpy array of shape (1400, 6)
    # Select channels 1 through 4 (in Python indexing, these are 0 through 3)
    selected_data = lsl_data[:, 0:4]

    # Transpose the data to make it 4x1400
    transformed_data = selected_data.T

    return transformed_data
def highpass_filter(data, cutoff, fs):
    b, a = butter(1, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)

def bandstop_filter(data, band, fs):
    b, a = butter(1, [band[0] / (0.5 * fs), band[1] / (0.5 * fs)], btype='bandstop')
    return filtfilt(b, a, data)

def filter_data(data, fs):
    filtered_data = np.copy(data)
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = highpass_filter(data[:, ch], 5, fs)
        filtered_data[:, ch] = bandstop_filter(filtered_data[:, ch], [58, 62], fs)
        filtered_data[:, ch] = bandstop_filter(filtered_data[:, ch], [118, 122], fs)
        filtered_data[:, ch] = bandstop_filter(filtered_data[:, ch], [178, 182], fs)
    return filtered_data


def waveform_length(data):
    waveform_lengths = np.zeros_like(data)
# Calculate waveform length for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        # Calculate the difference between consecutive measurements
        diffs = np.abs(np.diff(data[:, i], prepend=data[0, i]))

        # Sum the differences to get the waveform length for each measurement
        waveform_lengths[:, i] = np.cumsum(diffs)
    return waveform_lengths

def root_mean_square(data):
    rms_values = np.zeros_like(data)
    window_size = 50
    # Calculate RMS for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        for j in range(data.shape[0]):  # Iterate over measurements
            # Calculate the RMS for the current window ending at measurement j
            window = data[max(0, j - window_size + 1):j + 1, i]
            rms_values[j, i] = np.sqrt(np.mean(window**2))
    return rms_values
def mean_power(data, fs=1000):
    mean_power = np.zeros_like(data)

    # Calculate mean power for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        # Square the EMG signal values
        squared_signal = np.square(data[:, i])

        # Calculate the cumulative sum of the squared signal
        cumulative_sum = np.cumsum(squared_signal)

        # Calculate the mean power over time
        mean_power[:, i] = cumulative_sum / (np.arange(data.shape[0]) + 1)
    #print(mean_power)
    #print(np.shape(mean_power))
    return mean_power

def variance(data):
    variances = np.zeros_like(data)

    # Calculate variance for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        # Calculate the variance for the channel
        channel_variance = np.var(data[:, i])

        # Fill the variance array for the channel with the calculated variance
        variances[:, i].fill(channel_variance)
   
    return variances

def slope_sign_change(data):
    diff_data = np.diff(data, axis=1)
    return np.sum(np.multiply(diff_data[:, :-1], diff_data[:, 1:]) < 0, axis=1)

def simple_square_integral(data):
    squared_emg_data = np.square(data)

    # Initialize an array to store the SSI values
    ssi_values = np.zeros_like(data)

    # Calculate SSI for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        # Cumulative sum of squared values for each measurement
        ssi_values[:, i] = np.cumsum(squared_emg_data[:, i])
   
    return ssi_values


def median_frequency(data, fs, window_size = 50):
    median_frequencies = np.zeros_like(data)

    # Calculate median frequency over time for each channel
    for i in range(data.shape[1]):  # Iterate over channels
        for j in range(0, data.shape[0], window_size):
            # Select data in the current window
            window_data = data[j:j+window_size, i] if j + window_size <= data.shape[0] \
                          else data[j:, i]

            # Compute the power spectral density using Welch's method
            f, Pxx = welch(window_data, fs=fs, nperseg=min(len(window_data), 256))

            # Compute cumulative power spectrum
            cumulative_power = np.cumsum(Pxx)

            # Find the median frequency
            median_freq = f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]

            # Assign the median frequency to the measurements in this window
            end_index = j + window_size if j + window_size <= data.shape[0] else data.shape[0]
            median_frequencies[j:end_index, i] = median_freq
    #print(median_frequencies)
    #print(np.shape(median_frequencies))
    return median_frequencies

def extract_features(dataChTimeTr, included_features, fs):
    feature_table = {}
    
    num_channels = 4  # Assuming the first dimension is the number of channels

    for feature in included_features:
        if feature == 'WL':
            wl_values = waveform_length(dataChTimeTr)
            for ch in range(num_channels):
                print(np.shape(wl_values))
                feature_table[f'WL_{ch+1}'] = wl_values[:, ch]
        elif feature == 'RMS':
            rms_values = root_mean_square(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'RMS_{ch+1}'] = rms_values[:, ch]
        elif feature == 'MeanPower':
            mean_power_values = mean_power(dataChTimeTr, fs)
            for ch in range(num_channels):
                #print(np.shape(mean_power_values))
                feature_table[f'MeanPower_{ch+1}'] = mean_power_values[:, ch]
        elif feature == 'var':
            variance_val = variance(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'var_{ch+1}'] = variance_val[:, ch]
        elif feature == 'SSI':
            ssc_val = simple_square_integral(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'SSI_{ch+1}'] = ssc_val[:,ch]
        elif feature == 'MedianFreq':
            median_freq_val = median_frequency(dataChTimeTr, fs)
            for ch in range(num_channels):
                feature_table[f'MedianFreq_{ch+1}'] = median_freq_val[:,ch]
            #feature_table['MedianFreq'] = median_frequency(dataChTimeTr, fs)
        # Add more elif cases here for other features

    return feature_table

# Example usage
included_features = ['WL', 'RMS', 'MeanPower']
fs = 1000  # Sampling frequency
# feature_table = extract_features(dataChTimeTr, included_features, fs)
