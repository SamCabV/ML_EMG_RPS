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
        self.included_features =  ['WL', 'RMS', 'MeanPower', 'var', 'SSI', 'MedianFreq']
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

        transformed_data = transform_data(data)
        filtered_data = filter_data(transformed_data, fs)   
        feature_table = extract_features(filtered_data, self.included_features, self.fs)
        prediction = self.model.predict(feature_table)
        return self.tags[int(prediction[0])]
 

def transform_data(lsl_data):
    # Assuming lsl_data is a numpy array of shape (1400, 6)
    # Select channels 1 through 4 (in Python indexing, these are 0 through 3)
    selected_data = lsl_data[:, 1:4]

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
    # data shape is assumed to be (channels, time, trials)
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1)

def root_mean_square(data):
    return np.sqrt(np.mean(np.square(data), axis=1))

def mean_power(data, fs):
    num_trials = data.shape[2]
    num_channels = data.shape[0]
    fvalues = np.zeros((num_channels, num_trials))

    for ch in range(num_channels):
        for tr in range(num_trials):
            signal = data[ch, :, tr]
            power_spectrum = np.abs(fft(signal))**2 / len(signal)
            fvalues[ch, tr] = np.mean(power_spectrum)
    
    return fvalues.T  # Transposing to match the expected dimensions (trials, channels)


def variance(data):
    return np.var(data, axis=1)

def slope_sign_change(data):
    diff_data = np.diff(data, axis=1)
    return np.sum(np.multiply(diff_data[:, :-1], diff_data[:, 1:]) < 0, axis=1)

def simple_square_integral(data):
    return np.sum(np.square(data), axis=1)

def median_frequency(data, fs):
    num_trials = data.shape[2]
    num_channels = data.shape[0]
    fvalues = np.zeros((num_channels, num_trials))

    for ch in range(num_channels):
        for tr in range(num_trials):
            signal = data[ch, :, tr]
            power_spectrum = np.abs(fft(signal))**2 / len(signal)
            cumulative_sum = np.cumsum(power_spectrum)
            total_power = cumulative_sum[-1]
            median_index = np.where(cumulative_sum >= total_power / 2)[0][0]
            median_freq = (median_index - 1) * fs / len(signal)
            fvalues[ch, tr] = median_freq

    return fvalues.T

def extract_features(dataChTimeTr, included_features, fs):
    feature_table = {}
    num_channels = dataChTimeTr.shape[0]  # Assuming the first dimension is the number of channels

    for feature in included_features:
        if feature == 'WL':
            wl_values = waveform_length(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'WL_{ch+1}'] = wl_values[ch]
        elif feature == 'RMS':
            rms_values = root_mean_square(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'RMS_{ch+1}'] = rms_values[ch]
        elif feature == 'MeanPower':
            mean_power_values = mean_power(dataChTimeTr, fs)
            for ch in range(num_channels):
                feature_table[f'MeanPower_{ch+1}'] = mean_power_values[ch]
        elif feature == 'var':
            variance_val = variance(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'var_{ch+1}'] = variance_val[ch]
        elif feature == 'SSC':
            ssc_val = slope_sign_change(dataChTimeTr)
            for ch in range(num_channels):
                feature_table[f'SSC_{ch+1}'] = ssc_val[ch]
        elif feature == 'MedianFreq':
            median_freq_val = median_frequency(dataChTimeTr, fs)
            for ch in range(num_channels):
                feature_table[f'MedianFreq_{ch+1}'] = median_freq_val[ch]
            #feature_table['MedianFreq'] = median_frequency(dataChTimeTr, fs)
        # Add more elif cases here for other features

    return feature_table

# Example usage
included_features = ['WL', 'RMS', 'MeanPower']
fs = 1000  # Sampling frequency
# feature_table = extract_features(dataChTimeTr, included_features, fs)
