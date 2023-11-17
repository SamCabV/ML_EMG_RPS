clear all;
% Add path to required functions
addpath('C:\Users\sammy\Downloads\OfflineAnalysis');

% Specify the main data directory
main_data_dir = 'C:\Users\sammy\Downloads\OfflineAnalysis';

% Get all subfolders in the main data directory
subfolders = dir(main_data_dir);
subfolders = subfolders([subfolders.isdir]);  % Filter only directories

% Initialize variables for the final data and labels
all_dataChTimeTr = [];
all_labels = [];
counter = 0;
% Loop through each subfolder
for i = 1:length(subfolders)
    % Skip the '.' and '..' directories
    
    if strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
        continue;
    else
        counter = counter + 1;
    end

    % Construct path to the current subfolder
    current_folder = fullfile(main_data_dir, subfolders(i).name);

    % Get all .mat files in the current subfolder
    data_file_names = dir(fullfile(current_folder, 'lsl_data*.mat'));

    % Initialize variables for data and labels in the current folder
    dataChTimeTr = [];
    labels = [];

    % Process each file in the current folder
    for f = 1:length(data_file_names)
        load(fullfile(data_file_names(f).folder, data_file_names(f).name));
        [epochData, gesturelist] = preprocessData(lsl_data, marker_data);
        if size(epochData, 3) ~= length(gesturelist)
            error('Labels don''t match the trials');
        end
        dataChTimeTr = cat(3, dataChTimeTr, epochData);
        labels = [labels; gesturelist];
    end

    % Append current folder's data and labels to the final variables
    if counter == 1
        all_dataChTimeTr= dataChTimeTr;
        all_labels = labels;
    else
        all_dataChTimeTr = cat(3, all_dataChTimeTr, dataChTimeTr);
        all_labels = [all_labels; labels];
    end
   

    % Optionally, append current folder's data and labels to CSV files
    csvwrite(strcat(current_folder, '_rawData.csv'), dataChTimeTr);
    csvwrite(strcat(current_folder, '_labels.csv'), labels);
end

% Data augmentation: duplicate and add Gaussian noise
augmentedData = cat(3, all_dataChTimeTr, all_dataChTimeTr);
noiseMean = 0;
noiseVariance = 0.01; % Example variance
noise = noiseMean + sqrt(noiseVariance) * randn(size(all_dataChTimeTr));
augmentedData(:, :, size(all_dataChTimeTr, 3) + 1:end) = augmentedData(:, :, size(all_dataChTimeTr, 3) + 1:end) + noise;
all_labels = [all_labels ; all_labels];
all_dataChTimeTr = augmentedData;

% Save the final combined .mat file
save('allSubjFiles.mat', 'all_labels', 'all_dataChTimeTr');
csvwrite("all_lsl.csv", dataChTimeTr);
csvwrite("all_labels.csv", labels);
