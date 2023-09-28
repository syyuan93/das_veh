"""
Kalman Filter code is based on https://github.com/jingxiaoliu/TelecomTM
"""

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import sys
import os
sys.path.append("/sep/syyuan/projects")
from modules.car_tracking_utils import likelihood_1d, remove_unrealistic_tracking, plot_data, interp_nan_value
from collections.abc import Iterable
import time


class KF_tracking:

    def __init__(self, data, t_axis, x_axis, args):
        self.data = data
        self.t_axis = t_axis
        self.x_axis = x_axis
        self.dx = x_axis[1] - x_axis[0]
        self.args = args

    def detect_in_one_section(self, start_x, nx=15, detection_args=None, sigma=0.1, pclip=98, show_plot=False, plt_xlim=1000):
        if not detection_args:
            detection_args = self.args["detect"]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominenceWindow = detection_args["prominenceWindow"]

        peak_erode = np.zeros(len(self.t_axis))

        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))
        # a quick detection using the first 15 channels
        for i in range(nx):
            das_for_dt = self.data[start_x_idx+i]
            peak_loc = find_peaks(das_for_dt,
                                  prominence=minprominence,
                                  wlen=prominenceWindow,
                                  distance=minseparation)[0]
                                  # height=height)[0]
            peak_erode_tmp = likelihood_1d(peak_loc, self.t_axis, sigma)
            peak_erode += peak_erode_tmp

        peak_loc_tmp, _ = find_peaks(peak_erode, height=max(peak_erode) * 0., distance=minseparation)
        vmax = np.percentile(np.abs(self.data), pclip)

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios':[3,1]}, sharey=True)
            axes[0].imshow(self.data.T,
                           aspect="auto",
                           extent=[self.x_axis[0], self.x_axis[-1], self.t_axis[-1], self.t_axis[0]],
                           cmap="gray",
                           vmax=vmax,
                           vmin=-vmax)
            axes[0].axvline(x=self.x_axis[start_x_idx], c='r')
            axes[0].axvline(x=self.x_axis[start_x_idx+nx], c='b')
            axes[1].plot(peak_erode, self.t_axis, 'b')
            axes[1].plot(peak_erode[peak_loc_tmp], self.t_axis[peak_loc_tmp], 'r^')
            axes[0].plot([self.x_axis[start_x_idx+nx//2]] * len(peak_loc_tmp), self.t_axis[peak_loc_tmp], 'r^')
            axes[0].set_xlim([self.x_axis[0], plt_xlim])

        veh_base = peak_loc_tmp
        return veh_base

    def tracking_with_veh_base(self, start_x, end_x, veh_base, sigma_a=0.01, detection_args=None):
        if detection_args is None:
            detection_args = self.args['detect']

        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))
        end_x_idx = np.argmin(np.abs(end_x - self.x_axis))

        x_axis = self.x_axis[start_x_idx: end_x_idx + 1]

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominenceWindow = detection_args["prominenceWindow"]
        height = detection_args.get("height", None)

        veh_states = np.empty((len(veh_base), end_x_idx - start_x_idx + 1))
        veh_states[:] = np.nan
        veh_states_v = np.empty((len(veh_base), end_x_idx - start_x_idx + 1))
        veh_states_v[:] = np.nan

        R = 1
        Tk1k = np.empty((2, len(veh_base)))
        Tk1k[:] = np.nan
        Tkk = np.empty((2, len(veh_base)))
        Tkk[:] = np.nan
        Pkk = np.empty((2, 2, len(veh_base)))
        Pkk[:] = np.nan
        Pk1k = np.empty((2, 2, len(veh_base)))
        Pk1k[:] = np.nan
        Xv = np.empty(len(veh_base))
        Xv[:] = np.nan
        C = np.array([1, 0])
        veh_base_state = veh_base.copy()
        st = time.time()

        factor = 3

        for i in range(start_x_idx, end_x_idx + 1, factor):
            for v in range(len(veh_base)):

                if sum(~np.isnan(veh_states[v, :])) == 1:
                    Tkk[:, v] = [veh_states[v, ~np.isnan(veh_states[v, :])], 0]
                    # raise Exception
                    Xv[v] = x_axis[~np.isnan(veh_states[v, :])]
                    Pkk[:, :, v] = np.array([[0, 0], [0, 0]])
                    veh_base_state[v] = veh_base[v]
                elif sum(~np.isnan(veh_states[v, :])) == 0:
                    veh_base_state[v] = veh_base[v]
                else:
                    delta_x = self.x_axis[i] - Xv[v]
                    A = [[1, delta_x], [0, 1]]
                    Q = sigma_a * np.array([[0.25 * delta_x ** 4, 0.5 * delta_x ** 3], [0.5 * delta_x ** 3, delta_x ** 2]])
                    Tk1k[:, v] = np.matmul(A, Tkk[:, v])

                    Pk1k[:, :, v] = np.matmul(np.matmul(A, Pkk[:, :, v]), np.transpose(A)) + Q
                    veh_base_state[v] = Tk1k[0, v]

            das_for_dt = self.data[i]
            peak_loc, _ = find_peaks(das_for_dt, prominence=minprominence, wlen=prominenceWindow, distance=minseparation) #height=height)

            for p in range(len(veh_base_state)):
                dist_tmp = peak_loc - veh_base_state[p]
                idx_tmp = np.where((dist_tmp > -15) & (dist_tmp <= 30))[0]
                valid_tmp = dist_tmp[idx_tmp]
                valid_tmp_pos = valid_tmp[valid_tmp > 0]
                if len(valid_tmp_pos) == 0:
                    min_idx = []
                else:
                    min_idx = np.where(valid_tmp_pos == valid_tmp_pos.min())

                if len(min_idx) > 0:
                    veh_states[p, i - start_x_idx] = peak_loc[idx_tmp[min_idx]]
                elif len(valid_tmp) > 0:
                    valid_tmp_abs = np.abs(valid_tmp)
                    min_idx = np.where(valid_tmp_abs == valid_tmp_abs.min())
                    veh_states[p, i - start_x_idx] = peak_loc[idx_tmp[min_idx]]
                else:
                    veh_states[p, i - start_x_idx] = np.nan

            # filtering
            for v in range(len(veh_base)):

                if (sum(~np.isnan(veh_states[v, :])) > 2) and (not np.isnan(veh_states[v, i - start_x_idx])):
                    K = Pk1k[:, :, v] @ C.T / (R + C @ Pk1k[:, :, v] @ C.T)

                    # if i + 1 == 266 and v + 1 == 1:
                    #     print(Pk1k[:, :, v])
                    tkk = Tk1k[:, v] + K * (veh_states[v, i - start_x_idx] - C @ Tk1k[:, v])
                    # print("tkk",tkk.shape)
                    Tkk[:, v] = tkk
                    Pkk[:, :, v] = Pk1k[:, :, v] - (K.reshape(2, 1) @ C.reshape(1, 2)) @ Pk1k[:, :, v]
                    Xv[v] = self.x_axis[i]


            # if i + 1 == 280:
            #     print(tkk)
        tracked_v = remove_unrealistic_tracking(veh_base, veh_states, factor=factor)
        # tracked_v = veh_states
        tracked_v_full = np.empty((tracked_v.shape[0], tracked_v.shape[-1] * factor))
        tracked_v_full[:] = np.nan
        tracked_v_full[:, ::factor] = tracked_v
        # tracked_v = veh_states
        interp_nan_value(tracked_v_full)

        return tracked_v_full

    def tracking_visulization_one_section(self, start_x, tracked_v, plt_xlim=800, plt_tlim=78, t_min=0, ax=None,
                                          plot_tracking=True, plt_xlo=0, fontsize=16, tickfont=12, fig_dir=None, fig_name=None):
        start_x_idx = np.argmin(np.abs(start_x - self.x_axis))
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        self.plot_data(ax=ax)

        if plot_tracking:
            for v in range(len(tracked_v[:, 1])):
                tmp = tracked_v[v][~np.isnan(tracked_v[v, :])].astype(int)
                dist_idx_tmp = np.where(~np.isnan(tracked_v[v, :]))[0] + start_x_idx
                ax.plot(self.x_axis[dist_idx_tmp], self.t_axis[tmp], '.', color='red', markersize=1)

        ax.set_xlim([plt_xlo, plt_xlim])
        ax.set_ylim([plt_tlim, t_min])
        ax.set_xlabel("Distance along fiber [m]", fontsize=fontsize)
        ax.set_ylabel("Time [s]", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tickfont)
        if fig_dir and fig_name:
            fig_path = os.path.join(fig_dir, fig_name)
            plt.savefig(fig_path)
            print(f'{fig_path} is saved...')

    def plot_data(self, pclip=98, ax=None):
        x_axis, t_axis = self.x_axis, self.t_axis
        plot_data(self.data, x_axis, t_axis, pclip=pclip, ax=ax)

    def show_detection_example(self, x0=720, detection_args=None, pclip=98):
        if not detection_args:
            detection_args = self.args["detect"]

        if not isinstance(x0, Iterable):
            x0 = [x0]

        markers = ['r^', 'bv']

        minprominence = detection_args["minprominence"]
        minseparation = detection_args["minseparation"]
        prominenceWindow = detection_args["prominenceWindow"]
        height = detection_args.get("height", None)

        fig, axes = plt.subplots(1, 1+len(x0), gridspec_kw={'width_ratios': [3] + [1] * len(x0)}, sharey=True, figsize=(5*(2+len(x0)), 10))
        vmax = np.percentile(np.abs(self.data), pclip)

        axes[0].imshow(self.data.T,
                       aspect="auto",
                       extent=[self.x_axis[0], self.x_axis[-1], self.t_axis[-1], self.t_axis[0]],
                       cmap="gray",
                       vmax=vmax,
                       vmin=-vmax)

        for k, (x, m) in enumerate(zip(x0, markers)):
            x_idx = np.argmin(np.abs(x - self.x_axis))
            das_for_dt = self.data[x_idx]
            peak_loc, _ = find_peaks(das_for_dt, prominence=minprominence,
                                     distance=minseparation, wlen=prominenceWindow, height=height)

            axes[0].plot([self.x_axis[x_idx]]*len(peak_loc), self.t_axis[peak_loc], m)

            axes[k+1].plot(das_for_dt, self.t_axis, 'k')
            axes[k+1].plot(das_for_dt[peak_loc], self.t_axis[peak_loc], m)
            axes[k+1].axvline(x=minprominence, linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Telecom fiber response')
            plt.legend(['Response', 'Vehicle arrival'], loc='lower right')

        axes[0].set_xlim([self.x_axis[0], 1000])
        plt.show()







