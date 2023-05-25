from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import sys
import time

from modules.utils import bandpass_data_space, bandpass_data, plot_data, find_noise_idx, impute_noisy_trace
from apis.data_classes import SurfaceWaveSelector
from apis.imaging_classes import DispersionImagesFromWindows, VirtualShotGathersFromWindows
from apis.tracking import KF_tracking

channel_prop = {
    "odh3": {
        "start_ch": 400,  # start channel of the Sand Hill fiber
        "dx": 8.16
    }
}

# need at least one more class to feed in time windows into the TimeLapseImaging obj.
class TimeLapseImaging:
    def __init__(self, data, x_axis, t_axis, interrogator='odh3', method='surface_wave', tracking_preprecessing_dict=None, surface_wave_preprecessing_dict=None):
        """
        :param data: raw full-bandwidth DAS data;
                    for tracking, data need to be bandpassed, subsampling, and interpolation for channel axis
        :param x_axis: channel axis
        :param t_axis:
        :param start_x: start channel for tracking
        :param end_x: end channel for tracking
        :param tracking_args:
        """
        assert method in {'surface_wave', 'xcorr'}
        self.method = method
        channel_prop_ = channel_prop[interrogator]
        self.data = data
        self.t_axis = t_axis
        self.dt = self.t_axis[1] - self.t_axis[0]
        self.x_axis = x_axis
        self.start_ch = channel_prop_["start_ch"]
        self.dx = channel_prop_["dx"]
        self.distances_along_fiber = (x_axis - self.start_ch) * self.dx
        self.tracking_preprecessing_dict = tracking_preprecessing_dict
        self.surface_wave_preprecessing_dict = surface_wave_preprecessing_dict
        if tracking_preprecessing_dict is None:
            self.tracking_preprecessing_dict = {}

        self._preprocess_for_tracking()
        self._preprocessing_for_surface_waves()

    def _preprocessing_for_surface_waves(self, impute_noise_traces=True, noise_threshold=5, impute_empty_traces=True):
        # bandpass and remove noise data points
        norm_surfacewave = self.method == 'surface_wave'
        self.data_for_imaging = self.data.copy()
        if self.surface_wave_preprecessing_dict is None:
            self.surface_wave_preprecessing_dict = {}
        flo = self.surface_wave_preprecessing_dict.get('flo', 1.2)
        fhi = self.surface_wave_preprecessing_dict.get('fhi', 30)

        bandpass_data(self.data_for_imaging, self.dt, flo, fhi)

        if impute_empty_traces:
            noisy_index = find_noise_idx(self.data_for_imaging, noise_threshold=noise_threshold, empty_tr=True)
            impute_noisy_trace(self.data_for_imaging, noisy_index)

        if impute_noise_traces:
            noisy_index = find_noise_idx(self.data_for_imaging, noise_threshold=noise_threshold, empty_tr=False)
            impute_noisy_trace(self.data_for_imaging, noisy_index)

        if norm_surfacewave:
            self.data_for_imaging /= np.linalg.norm(self.data_for_imaging, axis=-1, keepdims=True)


    def _preprocess_for_tracking(self, subsamp_factor=5, noise_level=10, channel0=400):
        dry_data_quasi_for_tracking = copy.deepcopy(self.data)
        means = np.median(np.abs(dry_data_quasi_for_tracking), axis=-1)
        dry_data_quasi_for_tracking[means > noise_level] *= 0

        noisy_index = find_noise_idx(dry_data_quasi_for_tracking, noise_threshold=30, empty_tr=True)
        impute_noisy_trace(dry_data_quasi_for_tracking, noisy_index)

        # temporal bandpass
        flo = self.tracking_preprecessing_dict.get("flo", 0.08)
        fhi = self.tracking_preprecessing_dict.get("fhi", 1)
        bandpass_data(dry_data_quasi_for_tracking, self.dt, flo, fhi)

        # convert from 250 sampling rate to 50
        dry_data_quasi_for_tracking = dry_data_quasi_for_tracking[:, ::subsamp_factor]

        # spatial interpolation from 8.16 m to 1 m
        data_for_tracking = signal.resample_poly(dry_data_quasi_for_tracking, 204, 25)
        dist_along_fiber_tracking = np.arange(data_for_tracking.shape[0]) + (self.x_axis[0] - channel0) * 8.16
        t_axis_tracking = self.t_axis[::subsamp_factor]

        # spatial bandpass
        flo_space = self.tracking_preprecessing_dict.get("flo_space", 0.006)
        fhi_space = self.tracking_preprecessing_dict.get("fhi_space", 0.04)
        bandpass_data_space(data_for_tracking, dist_along_fiber_tracking, flo_space, fhi_space)

        self.data_for_tracking = data_for_tracking
        self.dist_along_fiber_tracking = dist_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking

    def track_cars(self, start_x, end_x, tracking_args, show_plot=True, reverse_amp=True, plt_xlim=1000, sigma_a=0.01):
        self.start_x = start_x
        self.end_x = end_x
        self.tracking_args = tracking_args
        if reverse_amp:
            data = -self.data_for_tracking
        else:
            data = self.data_for_tracking
        self.tracking = KF_tracking(data=data, t_axis=self.t_axis_tracking,
                                    x_axis=self.dist_along_fiber_tracking, args=self.tracking_args)

        veh_base = self.tracking.detect_in_one_section(start_x=self.start_x, nx=15, sigma=0.08, 
                                                       show_plot=show_plot, plt_xlim=plt_xlim)
        st = time.time()
        self.veh_states = self.tracking.tracking_with_veh_base(start_x=self.start_x, end_x=self.end_x,
                                                               veh_base=veh_base, sigma_a=sigma_a)
#         print('Time taken for car tracking:', time.time() - st)


    def visualize_tracking_on_surface_waves(self, ax=None, pclip=98, plt_xlo=0, plt_xlim=800, plt_tlo=0, plt_tlim=78, full_band=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        x_axis, t_axis = self.distances_along_fiber, self.t_axis
        data_to_plot = self.data if full_band else self.data_for_imaging
        plot_data(data_to_plot, x_axis, t_axis, pclip=pclip, ax=ax)

        dist_along_fiber_tracking = self.dist_along_fiber_tracking
        t_axis_tracking = self.t_axis_tracking
        start_x_idx = np.argmin(np.abs(self.start_x - dist_along_fiber_tracking))
        for v in range(len(self.veh_states[:, 1])):
            tmp = self.veh_states[v][~np.isnan(self.veh_states[v, :])].astype(int)
            dist_idx_tmp = np.where(~np.isnan(self.veh_states[v, :]))[0] + start_x_idx
            ax.plot(dist_along_fiber_tracking[dist_idx_tmp], t_axis_tracking[tmp], '.', color='red', markersize=1)

        ax.set_xlim([plt_xlo, plt_xlim])
        ax.set_ylim([plt_tlim, plt_tlo])

        if hasattr(self, "sw_selector"):
            for window in self.sw_selector:
                window.plot_on_data(ax=ax, c='y')

    def visualize_tracking(self, plt_tlim=100, plt_xlim=500, t_min=0,
                           ax=None, plot_tracking=True, plot_windows=True, fig_name=None, fig_dir='./', **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        self.tracking.tracking_visulization_one_section(start_x=self.start_x, tracked_v=self.veh_states,
                                                        plt_xlim=plt_xlim, plt_tlim=plt_tlim, ax=ax,
                                                        t_min=t_min,
                                                        plot_tracking=plot_tracking, **kwargs)

        if hasattr(self, "sw_selector") and plot_windows:
            for window in self.sw_selector:
                window.plot_on_data(ax=ax, c='y')

        if fig_name:
            fig_path = os.path.join(fig_dir, fig_name)
            print(f'{fig_path} is saved...')
            plt.savefig(fig_path)
            plt.close()


    def select_surface_wave_windows(self, x0, **kwargs):
        """
        Based on the tracking
        :return:
        """
        self.sw_selector = SurfaceWaveSelector(
            self.data_for_imaging,
            self.distances_along_fiber,
            self.t_axis,
            x0,
            self.start_x,
            self.veh_states,
            self.dist_along_fiber_tracking,
            self.t_axis_tracking,
            **kwargs
        )

    def get_images(self, mute_offset=300, **imaging_kwargs):
        image_from_window_cls = DispersionImagesFromWindows if self.method == "surface_wave" else VirtualShotGathersFromWindows
        self.images = image_from_window_cls(self.sw_selector)
        self.images.get_images(mute_offset=mute_offset, **imaging_kwargs)

    def save_disp_images(self, **kwargs):
        self.images.save_images(**kwargs)

    def save_avg_disp_to_npz(self, *args, fdir='/net/brick5/scratch1/siyuan/time_lapse_disp_results', **kwargs):
        self.images.avg_image.save_to_npz(*args, fdir=fdir, **kwargs)
