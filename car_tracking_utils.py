import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass


def read_data(data_dir, data_name):
    data_path = os.path.join(data_dir, data_name)
    data_file = np.load(data_path)
    return data_file["data"], data_file["x_axis"], data_file["t_axis"]

def read_from_csv_name(data_dir, data_name):
    data_dir_w_name = os.path.join(data_dir, data_name)
    das_channel = np.genfromtxt(data_dir_w_name + '_x_axis.csv')
    das_time = np.genfromtxt(data_dir_w_name + '_t_axis.csv')
    data = np.genfromtxt(data_dir_w_name + '.csv', delimiter=' ')
    return data, das_channel, das_time


def likelihood_1d(peak_loc, das_time_ds, sigma):
    data_tmp_thrd = np.zeros(len(das_time_ds))
    for j in range(len(peak_loc)):
        data_tmp_thrd = data_tmp_thrd + norm.pdf(das_time_ds, loc=das_time_ds[peak_loc[j]], scale=sigma)

    return data_tmp_thrd

def interp_nan_value(veh_states):
    for k, state in enumerate(veh_states):
        # Find indices of non-NaN values
        non_nan_indices = np.where(~np.isnan(state))[0]
        # Generate array of indices
        indices = np.arange(len(state))
        # Replace NaN values with linearly interpolated values
        state[np.isnan(state)] = np.interp(np.isnan(state).nonzero()[0], non_nan_indices, state[non_nan_indices])


def remove_unrealistic_tracking(veh_base, veh_states, adjacency_nan_count_lim=20, factor=1):
    invalid_num_tmp = []
    veh_states = veh_states[:, ::factor]
    for v in range(len(veh_base)):
        tmp = veh_states[v][~np.isnan(veh_states[v])]

        nan_indices = np.where(np.isnan(veh_states[v]))[0]
        diffs = np.diff(nan_indices)
        adjacency_count = np.sum(diffs == 1)
        # print(f'adjacency_nan_count_lim: {adjacency_nan_count_lim}')
        # adjacency_count = 0

        # remove unrealistic cases
        if len(~np.isnan(tmp)) < 0.3 * len(veh_states[v]) or sum(
               np.convolve(np.diff(tmp), np.ones(20), mode='valid') <= -15) or abs(sum(np.diff(tmp))) < 30 * (
               len(tmp) / len(veh_states[v])) or adjacency_count >= adjacency_nan_count_lim:
        # if len(~np.isnan(tmp)) < 0.3 * len(veh_states[v]):

            invalid_num_tmp.append(v)

        tmp_idx = np.where(~np.isnan(veh_states[v]))[0]
        invalid_idx = np.where(abs(np.diff(tmp)) > 20)[0]
        veh_states[v][tmp_idx[invalid_idx + 1]] = np.nan

    valid_num_tmp = list(range(len(veh_base)))
    for v in invalid_num_tmp:
        valid_num_tmp.remove(v)
    tracked_v = veh_states[valid_num_tmp, :]
    return tracked_v

def plot_data(data, x_axis, t_axis, pclip=98, ax=None, figsize=(10, 10), y_lim=None, x_lim=None):
    vmax = np.percentile(np.abs(data), pclip)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data.T,
              aspect="auto",
              extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
              cmap="gray",
              vmax=vmax,
              vmin=-vmax)
    # ax.set_xlim([0, 800])
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)

def bandpass_data(data, t_axis, flo, fhi):
    dt = t_axis[1] - t_axis[0]
    sampling_rate = int(1 / dt)
    for k, ch in enumerate(data):
        data[k] = bandpass(ch, freqmin=flo, freqmax=fhi, df=sampling_rate, corners=4, zerophase=True)

def read_and_plot_npz(data_dir, data_name, bp_params=None, **plt_kwargs):
    data, x_axis, t_axis = read_data(data_dir, data_name)
    if bp_params:
        bandpass_data(data, t_axis, **bp_params)
    plot_data(data, x_axis, t_axis, **plt_kwargs)


