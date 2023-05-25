# Virtual shot gathering

import copy
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy import interpolate
import scipy

from apis.data_classes import SurfaceWaveWindow
from modules.utils import XCORR_vshot, plot_xcorr, Dispersion, XCORR_two_traces, extrap1d

def xcorr_two_traces_based_on_traj(data, t_axis, pivot_idx, f, end_idx, wlen, dt, nsamp, x_axis, delta_t=1, reverse=False):
    nch = abs(end_idx - pivot_idx) - 1
    if reverse:
        nch += 1
    XCF_out = np.zeros((nch, int(wlen // dt)))
    start_idx = min(pivot_idx, end_idx)
    end_idx = max(pivot_idx, end_idx)
    if reverse:
        start_idx -= 1
    for k, x_idx in enumerate(range(start_idx + 1, end_idx)):
        t = f(x_axis[x_idx])
        if reverse:
            t -= delta_t
        else:
            t += delta_t
        t_idx = np.argmax(t_axis >= t)
        if reverse:
            tr1 = data[pivot_idx, t_idx - nsamp: t_idx]
            tr2 = data[x_idx, t_idx - nsamp: t_idx]
        else:
            tr1 = data[pivot_idx, t_idx: t_idx + nsamp]
            tr2 = data[x_idx, t_idx:  t_idx + nsamp]

        if reverse:
            vs, vr = tr1, tr2
        else:
            vs, vr = tr2, tr1
        XCF_out[k] = XCORR_two_traces(vs, vr, wlen, dt, overlap_ratio=0.5)

    return XCF_out

def plot_psd_vs_offset(XCF_out, x_axis, t_axis, ax=None, fhi=20, figsize=(8, 8), pclip=98, log_scale=False,
                       x_max=200, x_min=0, fname=None, fdir='./', vmax=None, vmin=None):

    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis * -1
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    dt = t_axis[1] - t_axis[0]
    fs = int(1 / dt)

    freq, Pxx_den = scipy.signal.welch(XCF_out, fs, nperseg=256, nfft=1024)

    fhi_idx = np.argmax(freq >= fhi)
    spec = Pxx_den[:, :fhi_idx]
    if log_scale:
        spec = 10 * np.log10(spec)
    if not vmax:
        vmax = np.percentile(spec, pclip)
    if not vmin:
        vmin = np.percentile(spec, 100-pclip)

    print(vmax, vmin)

    x_max_idx = np.abs(x_max - x_axis).argmin()
    x_min_idx = np.abs(x_min - x_axis).argmin()
    min_idx = min(x_max_idx, x_min_idx)
    max_idx = max(x_max_idx, x_min_idx)
    spec = spec[min_idx: max_idx]

    # X, Y = np.meshgrid(freq[:fhi_idx], x_axis[x_max_idx: x_min_idx])
    # print(X.shape, Y.shape, spec.shape)
    # ax.contourf(Y, X, spec)

    ax.imshow(spec.T, extent=[x_axis[min_idx], x_axis[max_idx], freq[fhi_idx], freq[0]],
              cmap='jet', aspect='auto', vmax=vmax, vmin=vmin, interpolation='antialiased')
    ax.set_xlabel("Distance along the fiber [m]")
    ax.set_ylabel("Frequency [Hz]")

    if fname:
        fpath = os.path.join(fdir, fname)
        plt.savefig(fpath)
        print(f'{fpath} has been saved...')
        plt.close()
    else:
        plt.show()


def plot_spectrum_vs_offset(XCF_out, x_axis, t_axis, ax=None, fhi=20, figsize=(8, 8), fname=None, fdir='./'):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    Nt = XCF_out.shape[-1]
    dt = t_axis[1] - t_axis[0]
    freq = np.fft.fftfreq(Nt, d=dt)
    fhi_idx = np.argmax(freq >= fhi)
    spec = np.fft.fft(XCF_out, axis=-1)[:, :fhi_idx]
    ax.imshow(np.abs(spec).T, extent=[x_axis[0], x_axis[-1], freq[fhi_idx], freq[0]], cmap='jet', aspect='auto')
    ax.set_xlabel("Distance along the fiber [m]")
    ax.set_ylabel("Frequency [Hz]")
    if fname:
        fpath = os.path.join(fdir, fname)
        plt.savefig(fpath)
        print(f'{fpath} has been saved...')
        plt.close()
    else:
        plt.show()

def preprocessing_window(window, pivot, delta_t, start_x, end_x, time_window_to_xcorr):
    f = interpolate.interp1d(window.veh_state_x, window.veh_state_t, fill_value='extrapolate')
    # f = extrap1d(f)

    dt = window.t_axis[1] - window.t_axis[0]
    pivot_idx = np.argmax(window.x_axis >= pivot)
    pivot_t = f(pivot) + delta_t
    pivot_t_idx = np.argmax(window.t_axis >= pivot_t)

    start_x_idx = np.argmax(window.x_axis >= start_x)
    end_x_idx = np.abs(window.x_axis - end_x).argmin()

    nsamp = int(time_window_to_xcorr // dt)

    data = window.data / np.linalg.norm(window.data)
    return pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f


def post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=False):
    x_axis = window.x_axis[start_x_idx: end_x_idx] - window.x_axis[pivot_idx]
    nt = XCF_out.shape[-1]
    t_axis = (np.arange(nt) - (nt // 2)) * dt

    if norm:
        XCF_out /= np.linalg.norm(XCF_out, axis=-1, keepdims=True)

    if norm_amp:
        XCF_out /= np.amax(XCF_out[pivot_idx - start_x_idx])
    if not reverse:
        XCF_out = XCF_out[:, ::-1]

    return XCF_out, x_axis, t_axis


def construct_shot_gather_other_side(window: SurfaceWaveWindow, start_x=530, end_x=680, pivot=635, wlen=2, norm=True, norm_amp=True,
                          time_window_to_xcorr = 4, delta_t=1):
    pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f = preprocessing_window(window, pivot, -delta_t,
                                                                                              start_x, end_x,
                                                                                              time_window_to_xcorr)

    # xcorr with the channels from the sources to the right of the pivot
    XCF_out_right = XCORR_vshot(data[pivot_idx: end_x_idx, pivot_t_idx - nsamp:pivot_t_idx], ivs=0,
                          wlen=wlen, dt=dt, reverse=True)

    # xcorr with the channels to the left of the pivot
    XCF_out_left = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f, start_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t, reverse=True)
    # print('other side shape', XCF_out_left.shape, XCF_out_right.shape)

    XCF_out = np.concatenate((XCF_out_left, XCF_out_right), axis=0)

    return post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp, reverse=True)



def construct_shot_gather(window: SurfaceWaveWindow, start_x=530, end_x=680, pivot=635, wlen=2, norm=True, norm_amp=True,
                          time_window_to_xcorr = 4, delta_t=1):
    # assert window.muted_along_traj

    pivot_idx, pivot_t_idx, start_x_idx, end_x_idx, nsamp, data, dt, f = preprocessing_window(window, pivot, delta_t, start_x, end_x, time_window_to_xcorr)

    # xcorr with the channels to the left of the pivot
    XCF_out = XCORR_vshot(data[start_x_idx: pivot_idx + 1, pivot_t_idx:pivot_t_idx+nsamp], pivot_idx - start_x_idx, wlen=wlen, dt=dt)
    # xcorr with the channels to the right of the pivot up to the source
    XCF_out_right = xcorr_two_traces_based_on_traj(data, window.t_axis, pivot_idx, f, end_x_idx, wlen, dt, nsamp, window.x_axis, delta_t=delta_t)
    # print('normal side shape', XCF_out.shape, XCF_out_right.shape)

    XCF_out = np.concatenate((XCF_out, XCF_out_right), axis=0)


    return post_processing_XCF(window, pivot_idx, start_x_idx, end_x_idx, XCF_out, dt, norm, norm_amp)


class VirtualShotGather:
    def __init__(self, window: SurfaceWaveWindow, compute_xcorr=True, disp=None, include_other_side=False, *args, **kwargs):
        self.window = window
        self.disp = disp
        if compute_xcorr:
            self.XCF_out, self.x_axis, self.t_axis = construct_shot_gather(window, *args, **kwargs)
            if include_other_side:
                XCF_out_other_side, _, _ = construct_shot_gather_other_side(window, *args, **kwargs)
                ch_idx_to_stack = np.linalg.norm(XCF_out_other_side, axis=-1) > 0
                self.XCF_out[ch_idx_to_stack] = (self.XCF_out[ch_idx_to_stack] + XCF_out_other_side[ch_idx_to_stack]) / 2


    def __add__(self, other):
        sum_ = copy.deepcopy(self)
        length = min(self.XCF_out.shape[-1], other.XCF_out.shape[-1])
        sum_.XCF_out[:, :length] += other.XCF_out[:, :length]
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other):
        new_obj = copy.deepcopy(self)
        new_obj.XCF_out /= other
        return new_obj

    @classmethod
    def get_VirtualShotGather_obj(cls, fdir, fname):
        new_obj = cls(window=None, compute_xcorr=False)
        file = np.load(os.path.join(fdir, fname), allow_pickle=True)
        new_obj.XCF_out, new_obj.x_axis, new_obj.t_axis = file["XCF_out"], file["x_axis"], file["t_axis"]
        return new_obj

    def plot_spec_vs_offset(self, ax=None, psd=True, pclip=98, fdir='Fig/virtual_gathers', fname=None,
                            x_max=100, x_min=-100, log_scale=False,
                            vmin=None, vmax=None):
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        if not psd:
            plot_spectrum_vs_offset(self.XCF_out, self.x_axis, self.t_axis, ax=ax, fdir=fdir, fname=fname)
        else:
            plot_psd_vs_offset(self.XCF_out, self.x_axis, self.t_axis, ax=ax, pclip=pclip,
                               x_max=x_max, x_min=x_min, fdir=fdir, fname=fname, log_scale=log_scale,
                               vmax=vmax, vmin=vmin)

    def save_to_npz(self, fname, fdir, **kwargs):
        np.savez(os.path.join(fdir, fname), XCF_out=self.XCF_out, x_axis=self.x_axis, t_axis=self.t_axis, **kwargs)

    def plot_image(self, fig_name=None, fig_dir=None,  x_lim=None, norm=False, plot_disp=False, plot_kwargs={}, **kwargs):
        if x_lim is None:
            x_lim = [-200, 200]
        if not plot_disp:
            ax = kwargs.get('ax')
            if not ax:
                fig, ax = plt.subplots(figsize=(8, 10))
            plot_xcorr(self.XCF_out, self.t_axis, self.x_axis, ax=ax, fig_dir=fig_dir, fig_name=fig_name, x_lim=x_lim, **plot_kwargs)
        else:
            assert self.disp, "please run obj.compute_disp_image() first"
            self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)


    def compute_disp_image(self, freqs = np.arange(0.8, 25, 0.1), vels = np.arange(200, 1200), norm=False, start_x=None, end_x=None):
        if start_x is None:
            start_x = self.x_axis[0]
        if end_x is None:
            end_x = self.x_axis[-1]

        start_x_idx = np.abs(self.x_axis - start_x).argmin()
        end_x_idx = np.abs(self.x_axis - end_x).argmin()
        # start_x_idx = np.argmax(self.x_axis <= start_x)
        # end_x_idx = np.argmax(self.x_axis <= end_x)
        self.disp = Dispersion(self.XCF_out[start_x_idx: end_x_idx + 1], 8.16, self.t_axis[1] - self.t_axis[0],
                          freqs=freqs, vels=vels, norm=norm)

    def plot_disp(self, fig_name=None, fig_dir="Fig/dispersion/", norm=True, **kwargs):
        assert self.disp, "please run obj.compute_disp_image() first"
        self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)

    def save_disp_to_npz(self, *args, **kwargs):
        assert self.disp, "please run obj.compute_disp_image() first"
        self.disp.save_to_npz(*args, **kwargs)


    def norm(self):
        self.XCF_out /= np.linalg.norm(self.XCF_out, axis=-1, keepdims=True)








