# Class and functions for selecting surface wave windows
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
from scipy import signal

from modules.utils import plot_data, extrap1d

class SurfaceWaveWindow:
    def __init__(
            self,
            data,
            x_axis,
            t_axis,
            veh_state,
            start_x_tracking,
            distance_along_fiber_tracking,
            t_axis_tracking,
    ):
        self.data = data
        self.x_axis = x_axis
        self.t_axis = t_axis
        self.veh_state = veh_state
        self.start_x_tracking = start_x_tracking
        self.distance_along_fiber_tracking = distance_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking
        self.muted_along_traj = False
        self.muted_along_time = False
        self._preprocess_veh_state()

    def _preprocess_veh_state(self):
        tmp = self.veh_state[~np.isnan(self.veh_state)].astype(int)
        start_x_tracking_idx = np.abs(self.start_x_tracking - self.distance_along_fiber_tracking).argmin()
        dist_idx_tmp = np.where(~np.isnan(self.veh_state))[0] + start_x_tracking_idx
        self.veh_state_x = self.distance_along_fiber_tracking[dist_idx_tmp]
        self.veh_state_t = self.t_axis_tracking[tmp]

    def plot_on_data(self, ax, c='r'):
        x_axis, t_axis = self.x_axis, self.t_axis
        length_sw = x_axis[-1] - x_axis[0]
        wlen_sw = t_axis[-1] - t_axis[0]
        rect = patches.Rectangle((x_axis[0], t_axis[0]), length_sw, wlen_sw, linewidth=1,
                                 edgecolor=c, facecolor='none')
        ax.add_patch(rect)

    def mute_along_traj(self, offset=200, alpha=0.3, delta_x=20):
        '''
        Generate an another SurfaceWaveWindow obj.
        :param offset:
        :return:
        '''
        f = interpolate.interp1d(self.veh_state_t, self.veh_state_x, fill_value='extrapolate')
        car_positions = f(self.t_axis)
        dx = self.x_axis[1] - self.x_axis[0]
        nx = self.x_axis.size
        n_samp = int(offset / dx)
        for k, (t, car_loc) in enumerate(zip(self.t_axis, car_positions)):
            muting_window = np.zeros((nx, 1))
            center_x = car_loc - offset / 2 + delta_x
            center_x_idx = np.argmax(self.x_axis > center_x)
            start_idx = max(0, center_x_idx - n_samp // 2)
            end_idx = min(nx, center_x_idx + n_samp // 2)
            taper_start_idx = start_idx + n_samp // 2 - center_x_idx
            taper_end_idx = taper_start_idx + end_idx - start_idx
            muting_window[start_idx: end_idx] = signal.windows.tukey(n_samp, alpha).reshape(n_samp, 1)[
                                                taper_start_idx: taper_end_idx]
            self.data[:, k] *= muting_window.reshape((muting_window.size,))

        self.muted_along_traj = True

    def mute_along_traj_double_sided(self, offset=200, alpha=0.05, delta_x=20):
        '''
        Generate an another SurfaceWaveWindow obj.
        :param offset:
        :return:
        '''
        f = interpolate.interp1d(self.veh_state_t, self.veh_state_x, fill_value='extrapolate')
        f = extrap1d(f)
        car_positions = f(self.t_axis)
        dx = self.x_axis[1] - self.x_axis[0]
        nx = self.x_axis.size
        n_samp = int(offset / dx)
        for k, (t, car_loc) in enumerate(zip(self.t_axis, car_positions)):
            muting_window = np.zeros((nx, 1))
            center_x = car_loc
            center_x_idx = np.argmax(self.x_axis > center_x)
            start_idx = max(0, center_x_idx - n_samp // 2)
            end_idx = min(nx, center_x_idx + n_samp // 2)
            taper_start_idx = start_idx + n_samp // 2 - center_x_idx
            taper_end_idx = taper_start_idx + end_idx - start_idx
            muting_window[start_idx: end_idx] = signal.windows.tukey(n_samp, alpha).reshape(n_samp, 1)[
                                                taper_start_idx: taper_end_idx]
            self.data[:, k] *= muting_window.reshape((muting_window.size,))

        self.muted_along_traj = True

    def mute_along_time(self, alpha=0.3):
        n_samp = self.data.shape[-1]
        muting_window = signal.windows.tukey(n_samp, alpha).reshape(1, n_samp)
        self.data *= muting_window
        self.muted_along_time = True

    def save_fig(self, fig_name=None, fig_dir="Fig/show_sw_time_windows/", t_min=None, t_max=None, x_min=None, x_max=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.veh_state_x, self.veh_state_t, '.', color='red', markersize=1)
        t_min = t_min if t_min is not None else self.t_axis[0]
        t_max = t_max if t_max is not None else self.t_axis[-1]

        x_min = x_min if x_min is not None else self.x_axis[0]
        x_max = x_max if x_max is not None else self.x_axis[-1]

        t_min_idx = np.abs(t_min - self.t_axis).argmin()
        t_max_idx = np.abs(t_max - self.t_axis).argmin()

        x_min_idx = np.abs(x_min - self.x_axis).argmin()
        x_max_idx = np.abs(x_max - self.x_axis).argmin()

        plot_data(self.data[x_min_idx: x_max_idx + 1, t_min_idx: t_max_idx + 1], x_axis=self.x_axis[x_min_idx: x_max_idx + 1],
                  t_axis=self.t_axis[t_min_idx: t_max_idx + 1],
                  fig_dir=fig_dir, fig_name=fig_name, ax=ax)


class SurfaceWaveSelector:
    def __init__(self, data_for_surface_wave,
                 distances_along_fiber,
                 t_axis,
                 x0,
                 start_x_tracking,
                 veh_states,
                 distance_along_fiber_tracking,
                 t_axis_tracking,
                 wlen_sw=8,
                 length_sw=300,
                 spatial_ratio=0.75,
                 temporal_spacing=None
                 ):
        """

        :param data_for_surface_wave:
        :param distances_along_fiber: x axis for the surface-wave data
        :param t_axis: t axis for the surface-wave data
        :param x0: middle distance for the window

        :param start_x_tracking: start_x for the tracking
        :param veh_states: tracked veh states
        :param distance_along_fiber_tracking: x axis for the tracking data
        :param t_axis_tracking: t axis for the tracking data
        :param wlen_sw: wlen for the surface wave window
        :param length_sw: spatial distance for the surface wave window
        """
        self.data_for_surface_wave = data_for_surface_wave
        self.distances_along_fiber = distances_along_fiber
        self.t_axis = t_axis
        self.dt = self.t_axis[1] - self.t_axis[0]
        self.x0 = x0
        self.start_x_tracking = start_x_tracking
        self.veh_states = veh_states
        self.distance_along_fiber_tracking = distance_along_fiber_tracking
        self.t_axis_tracking = t_axis_tracking
        self.wlen_sw = wlen_sw
        self.length_sw = length_sw
        self.spatial_ratio = spatial_ratio
        self.temporal_spacing = temporal_spacing if temporal_spacing else self.wlen_sw

        self.locate_windows()

    def locate_windows(self):
        win_nsamp = int(self.wlen_sw / self.dt)
        x0_idx = self.x0 - self.start_x_tracking
        windows = []
        has_car_behind = []
        has_car_ahead = []
        for k, v in enumerate(self.veh_states):
            t0_idx = int(v[x0_idx])

            # reject cars behind it
            if k < len(self.veh_states) - 1:
                t0_next_v_idx = int(self.veh_states[k + 1, x0_idx])
                if self.t_axis_tracking[t0_next_v_idx] - self.t_axis_tracking[t0_idx] < self.temporal_spacing:
                    has_car_behind.append(k)
                    continue

            # reject cars ahead of it
            if k > 0:
                t0_before_v_idx = int(self.veh_states[k - 1, x0_idx])

                delta_t = self.t_axis_tracking[t0_idx] - self.t_axis_tracking[t0_before_v_idx]

                if self.temporal_spacing > delta_t > 0:
                    has_car_ahead.append(k)
                    continue

            t0 = self.t_axis_tracking[t0_idx]
            t0_sw_idx = np.abs(t0 - self.t_axis).argmin()
            
            # reject windows at the boundaries
            if t0_sw_idx < win_nsamp // 2 or t0_sw_idx + win_nsamp // 2 > self.t_axis.size:
                continue

            start_x = self.x0 - self.length_sw * self.spatial_ratio
            end_x = start_x + self.length_sw

            start_x_idx = np.abs(start_x - self.distances_along_fiber).argmin()
            end_x_idx = np.abs(end_x - self.distances_along_fiber).argmin()

            start_t0_idx = t0_sw_idx - win_nsamp // 2
            end_t0_idx = start_t0_idx + win_nsamp

            sw_window = SurfaceWaveWindow(
                data=copy.deepcopy(self.data_for_surface_wave[start_x_idx: end_x_idx, start_t0_idx: end_t0_idx]),
                t_axis=self.t_axis[start_t0_idx: end_t0_idx],
                x_axis=self.distances_along_fiber[start_x_idx: end_x_idx],
                veh_state=v,
                start_x_tracking=self.start_x_tracking,
                distance_along_fiber_tracking=self.distance_along_fiber_tracking,
                t_axis_tracking=self.t_axis_tracking,
            )

            windows.append(sw_window)

        self.windows = windows

    # return number of isolated cars
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        return self.windows[item]

    def __setitem__(self, key, value):
        self.windows[key] = value

    def __contains__(self, item):
        return item >= 0 and item < len(self.windows)

    def overlay_windows_on_data(self, data=None):
        if data is None:
            data = self.data_for_surface_wave
        fig, ax = plt.subplots()
        plot_data_by_meter(data, self.distances_along_fiber, self.t_axis, ax=ax)
        for window in self.windows:
            window.plot_on_data(ax, c='y')

    def save_figs(self, muted=False, offset=450, fig_dir="Fig/show_sw_time_windows/"):
        for k, win in enumerate(self):
            fig_prefix = 'sw_car'
            if muted:
                win_to_save = copy.deepcopy(win)
                win_to_save.mute_along_traj(offset=offset, alpha=0.6)
                fig_prefix += '_muted'
            else:
                win_to_save = win
            win_to_save.save_fig(fig_name=f"{fig_prefix}{k}.png", fig_dir=fig_dir)

