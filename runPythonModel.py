import numpy as np
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt
import joblib

# Load the model from file


class RunPythonModel:
    def __init__(self, modelPath):
        self.model = None # replace this with whatever code you need to load model
        # Example usage
        self.included_features =  ['WL', 'RMS', 'MeanPower', 'var', 'SSC', 'MedianFreq']
        self.fs = 1000  # Sampling frequency
        self.model_file_name = 'my_random_forest_model.joblib'
        self.model = joblib.load(self.model_file_name)
        self.tags = {0: 'rock', 1: 'paper', 2: 'scissors', 3: 'Resting'}
        self.featur_order = ['var', 'WL', 'RMS', 'SSC', 'MeanPower', "MedianFreq"]
    def get_rps(self,data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """

        #print(np.shape(data))
        transformed_data = transform_data(data)
        #print(np.shape(transformed_data))
        filtered_data = filter_data(transformed_data, fs) 
        filtered_data = filtered_data.T
        feature_table = extract_features(filtered_data, self.included_features, self.fs)
        print(feature_table)
        feature_table['MeanPower'] = feature_table['MeanPower'].flatten()
        feature_table['MedianFreq'] = feature_table['MedianFreq'].flatten()

        matrix = np.column_stack([feature_table[feature] for feature in self.featur_order])
        print(np.shape(matrix))
        prediction = self.model.predict(matrix)
        return self.tags[int(prediction[0])]
 

def transform_data(lsl_data):
    # Assuming lsl_data is a numpy array of shape (1400, 6)
    # Select channels 1 through 4 (in Python indexing, these are 0 through 3)
    selected_data = lsl_data[:, 1:5]

    # Transpose the data to make it 4x1400
    #transformed_data = selected_data.T
    transformed_data = selected_data
    
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
    # Assuming data is a 2D array of shape (trials, channels)
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)

def root_mean_square(data):
    # Assuming data is a 2D array of shape (trials, channels)
    return np.sqrt(np.mean(np.square(data), axis=0))


def mean_power(data, fs):
    num_channels = data.shape[1]  # Number of channels
    power_values = np.zeros(num_channels)

    for ch in range(num_channels):
        signal = data[:, ch]
        if signal.size == 0:
            continue
        power_spectrum = np.abs(fft(signal))**2 / len(signal)
        power_values[ch] = np.mean(power_spectrum)
    
    return power_values



# Assuming dataChTimeTr is a 3D NumPy array (channels x timepoints x trials)
# and included_features is a list of strings indicating which features to extract
def variance(data):
    # Assuming data is a 2D array of shape (trials, channels)
    return np.var(data, axis=0)

def slope_sign_change(data):
    diff_data = np.diff(data, axis=0)
    return np.sum(np.multiply(diff_data[:-1, :], diff_data[1:, :]) < 0, axis=0)

def simple_square_integral(data):
    return np.sum(np.square(data), axis=1)

def median_frequency(data, fs):
    num_channels = data.shape[1]
    median_freq_values = np.zeros(num_channels)

    for ch in range(num_channels):
        signal = data[:, ch]
        if signal.size == 0:
            continue
        power_spectrum = np.abs(fft(signal))**2 / len(signal)
        cumulative_sum = np.cumsum(power_spectrum)
        total_power = cumulative_sum[-1]
        median_index = np.where(cumulative_sum >= total_power / 2)[0][0]
        median_freq = (median_index - 1) * fs / len(signal)
        median_freq_values[ch] = median_freq
    
    return median_freq_values



def extract_features(data, included_features, fs):
    feature_table = {}

    for feature in included_features:
        if feature == 'WL':
            feature_table['WL'] = waveform_length(data)
        elif feature == 'RMS':
            feature_table['RMS'] = root_mean_square(data)
        elif feature == 'MeanPower':
            feature_table['MeanPower'] = mean_power(data, fs)
        elif feature == 'var':
            feature_table['var'] = variance(data)
        elif feature == 'SSC':
            feature_table['SSC'] = slope_sign_change(data)
        elif feature == 'MedianFreq':
            feature_table['MedianFreq'] = median_frequency(data, fs)

    return feature_table

# Example usage
included_features = ['WL', 'RMS', 'MeanPower']
fs = 1000  # Sampling frequency
# feature_table = extract_features(dataChTimeTr, included_features, fs)
