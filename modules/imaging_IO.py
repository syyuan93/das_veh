# The ImagingIO class gets the file list in data directory

import os
from modules.utils import _read_das_npz
from datetime import datetime
from scipy.signal import savgol_filter

def get_file_list(directory):
    data_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory, file_name)
            data_files.append((file_path, file_name))
    data_files.sort(key=lambda x: x[1])
    return [x[0] for x in data_files]

def get_time_from_file_path(file_path, time_format='%Y%m%d_%H%M%S'):
    file_name = os.path.basename(file_path)
    time_str = file_name.split('.')[0]
    return datetime.strptime(time_str, time_format)

# TODO: able to select time window based on the starttime
class ImagingIO:
    def __init__(self, directory, root, ch1=400, ch2=540, smoothing=True):
        folder_path = os.path.join(root, directory)
        self.data_files = get_file_list(folder_path)
        self.ch1 = ch1
        self.ch2 = ch2
        self.smoothing = smoothing

    def get_time_interval(self):
        start_time = get_time_from_file_path(self.data_files[0])
        end_time = get_time_from_file_path(self.data_files[1])
        interval = (end_time - start_time).total_seconds()
        return interval

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data, x_axis, t_axis = _read_das_npz(file_path, ch1=self.ch1, ch2=self.ch2)
        scale = 1
        date = file_path.split('/')[-2]
        if date > '20230219':
            scale = 6463.81735715902
        if self.smoothing:
            data = savgol_filter(data, 21, 15)
        data /= scale

        return data, x_axis, t_axis

    def __contains__(self, item):
        return 0 < item < len(self.data_files)

    def __len__(self):
        return len(self.data_files)










