import sys, os
import copy
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.filter import bandpass
import segyio
from scipy import signal
import math
import scipy
import time
import datetime


from scipy.interpolate import interp1d
from scipy import arange, array, exp


def pick_travel_time(vs, freq=12, dt = 1 / 250, nptsfreq=200, vpo=12):
    from xwt import cwt

    travel_times = []
    for k, tr in enumerate(vs.XCF_out):
        cc_cwt, cc_cfs, _freqs, coi = cwt(tr, 1 / dt, freqmin=2, freqmax=12, smooth=True, vpo=vpo, nptsfreq=nptsfreq)
        cc_cfs = cc_cfs[:, cc_cfs.shape[1]//2:]
        max_idx = np.argmax(cc_cfs, axis=-1)
        freq_idx = np.abs(freq - _freqs).argmin()
        max_idx = max_idx[freq_idx]
        if k == 0:
            print(_freqs[freq_idx])
        travel_times.append(max_idx * dt)
    return travel_times



def lcm(x, y):
   """This function takes two
   integers and returns the L.C.M."""

   # Choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike


def _read_das_segy(fname, **kwargs):
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb

    with segyio.open(fname, ignore_geometry=True) as segy_file:
        nch = segy_file.tracecount
        nt = segy_file.samples.size
        dt = segyio.tools.dt(segy_file) / 1e6

        ch1 = kwargs.get('ch1', 0)
        ch2 = kwargs.get('ch2', nch)

        data = segy_file.trace.raw[ch1:ch2]

        return data, np.arange(ch1, ch2), np.arange(nt) * dt

def _cut_taper(data, t_axis):
    nt = data.shape[-1]
    taper_len = np.argmin(np.abs(t_axis))
    t_axis = t_axis[taper_len: nt - taper_len]
    data = data[:, taper_len: nt - taper_len]
    return data, t_axis

def _read_das_npz(fname, **kwargs):
    try:
        data_file = np.load(fname)
    except:
        raise Exception(f"fname: {fname}")
    data = data_file["data"]
    x_axis = data_file["x_axis"]

    t_axis = data_file["t_axis"]
    ch1 = kwargs.get('ch1', x_axis[0])
    ch2 = kwargs.get('ch2', x_axis[-1])

    ch1_idx = np.argmax(x_axis >= ch1)
    ch2_idx = np.argmax(x_axis >= ch2)
    data = data[ch1_idx:ch2_idx]

    cut_taper = kwargs.get("cut_taper", True)
    if cut_taper:
        data, t_axis = _cut_taper(data, t_axis)
    return data, x_axis[ch1_idx:ch2_idx], t_axis


file_reading_func = {
    ".segy": _read_das_segy,
    ".npz": _read_das_npz
}

def das_preprocess(data_in):
    data_out = signal.detrend(data_in)
    data_out = data_out - np.median(data_out, axis=0)
    return data_out

def taper_data(data):
    taper = signal.tukey(data.shape[1], 0.05)
    taper = taper.reshape((1, taper.size))
    data *= taper

def cut_data_along_time(data, t_axis, t1, t2):
    t1_idx = np.abs(t1 - t_axis).argmin()
    t2_idx = np.abs(t2 - t_axis).argmin()
    return data[:, t1_idx: t2_idx], t_axis[t1_idx:t2_idx]

def read_das_files(fnames, bp_params, preprocess=False, **kwargs):
    if not isinstance(fnames, list):
        fnames = [fnames]

    data = []
    t_axis = []
    t_shift = 0
    for k, fname in enumerate(fnames):
        file_suffix = os.path.splitext(fname)[-1]
        io_func = file_reading_func[file_suffix]
        d, x, t = io_func(fname, **kwargs)
        dt = t[1] - t[0]
        nt = t.size
        data.append(d)
        t_axis.append(t + t_shift)
        t_shift += nt * dt
    data = np.concatenate(data, axis=-1)
    t_axis = np.concatenate(t_axis)
    x_axis = x

    if preprocess or (preprocess is None and file_suffix == ".segy"):
        print("preprocessing...")
        data = das_preprocess(data)

    if bp_params:
        taper_data(data)
        bandpass_data(data, dt, **bp_params)

    data, t_axis = cut_data_along_time(data, t_axis=t_axis, t1=kwargs.get("t1", 0), t2=kwargs.get("t2", t_axis[-1]))

    return data, x_axis, t_axis


def read_data(data_dir, data_name, bp_params, preprocess=None, **kwargs):
    if not isinstance(data_name, list):
        data_name = [data_name]
    data_paths = []
    for d_n in data_name:
        data_paths.append(os.path.join(data_dir, d_n))

    return read_das_files(data_paths, bp_params=bp_params, preprocess=preprocess, **kwargs)


def bandpass_data(data, dt, flo, fhi):
    sampling_rate = int(1 / dt)

    fNy = 0.5 / dt
    order = 10
    # st = time.time()
    # sos = signal.butter(order, [flo / fNy, fhi / fNy], analog=False, btype='band')
    sos = signal.butter(order, [flo / fNy, fhi / fNy], analog=False, btype='band', output='sos')
    data[:] = signal.sosfiltfilt(sos, data, axis=1)
    # print(data)
    # data[:] = signal.sosfilt(sos, data, axis=1)
    # print('old...',time.time()-st)

    # st = time.time()
    # for k, ch in enumerate(data):
    #     data[k] = bandpass(ch, freqmin=flo, freqmax=fhi, df=sampling_rate, corners=4, zerophase=True)
    # print('new...', time.time()-st)


def plot_data(data, x_axis, t_axis, pclip=98, ax=None, figsize=(10, 10), y_lim=None, x_lim=None, fig_name=None, fig_dir="Fig/", fontsize=16, tickfont=12):
    vmax = np.percentile(np.abs(data), pclip)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data.T,
              aspect="auto",
              extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
              cmap="gray",
              vmax=vmax,
              vmin=-vmax)
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    plt.xlabel("Distance along the fiber [m]", fontsize=fontsize)
    ax.set_ylabel("Time [s]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfont)
    if fig_name:
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

def read_and_plot_npz(data_dir, data_name, read_params=None, bp_params=None, return_data=False, preprocess=None, **plt_kwargs):
    data, x_axis, t_axis = read_data(data_dir, data_name, bp_params, preprocess=preprocess, **read_params)
    plot_data(data, x_axis, t_axis, **plt_kwargs)
    if return_data:
        return data, x_axis, t_axis

def compute_and_plot_fk(data, dx, dt):
    fk_res, fft_f, fft_k = fk(data, dx, dt)
    plot_fk(fk_res, fft_f, fft_k)

def plot_fk(fk_res, fft_f, fft_k):
    plt.figure(figsize=(10, 10))
    plt.imshow(fk_res.T, extent=[fft_k[0], fft_k[-1], fft_f[-1], fft_f[0]], aspect="auto")
    plt.ylim([0, 20])
    plt.xlim([0, 0.04])
    plt.show()

def fk(data, dx, dt):
    # From Ariel's repo
    (nch, nt) = np.shape(data)
    nf = 2 ** (1 + math.ceil(math.log(nt, 2)))
    nk = 2 ** (1 + math.ceil(math.log(nch, 2)))

    fft_f = np.arange(-nf / 2, nf / 2) / nf / dt
    fft_k = np.arange(-nk / 2, nk / 2) / nk / dx

    fk_res = np.fft.fftshift(np.fft.fft2(data, s=[nk, nf]))
    fk_res = np.absolute(fk_res)

    return fk_res, fft_f, fft_k

def repeat1d(trace):
    return np.hstack((trace, trace[:-1]))

def XCORR_two_traces(tr1, tr2, wlen, dt, overlap_ratio=0.5):
    nt = tr1.size
    wlen = int(wlen / dt)
    wlen_offset = int(wlen * (1 - overlap_ratio))
    nwin = (nt - wlen) // wlen_offset + 1

    XCF_out = np.zeros((1, wlen))

    for iwin in range(nwin):
        data_vs = repeat1d(tr1[(iwin * wlen_offset):(iwin * wlen_offset) + wlen])
        data_vr = tr2[(iwin * wlen_offset):(iwin * wlen_offset) + wlen]

        XCF_out += np.asarray(signal.correlate(data_vs, data_vr,
                                                mode='valid', method='fft'))
    XCF_out = np.roll(XCF_out, wlen // 2, axis=-1)
    if nwin > 0:
        XCF_out /= nwin
    return XCF_out

def get_date_string_list(start_date_str, end_date_str, date_fmt='%Y%m%d'):

    # Create datetime objects from the input dates
    start_date = datetime.datetime.strptime(start_date_str, date_fmt)
    end_date = datetime.datetime.strptime(end_date_str, date_fmt)

    # Initialize an empty list to store the dates
    date_list = []

    # Loop through the dates between start and end dates and append to the list
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime(date_fmt))
        current_date += datetime.timedelta(days=1)

    return date_list

def XCORR_vshot(data, ivs, wlen, dt, overlap_ratio=0.5, reverse=False):
    nch, nt = data.shape[0], data.shape[1]

    wlen = int(wlen / dt)

    wlen_offset = int(wlen * (1 - overlap_ratio))
    nwin = (nt - wlen) // wlen_offset + 1

    XCF_out = np.zeros((nch, wlen))
    for iwin in range(nwin):
        data_vs = repeat1d(data[ivs, (iwin * wlen_offset):(iwin * wlen_offset) + wlen])
        curt_XCF = []
        for ivr in range(nch):
            if reverse:
                vs = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
                vr = data_vs
            else:
                vs = data_vs
                vr = data[ivr, (iwin * wlen_offset):(iwin * wlen_offset) + wlen]
            curt_XCF.append(signal.correlate(vs, vr, mode='valid', method='fft'))

        XCF_out += np.asarray(curt_XCF)
    if nwin == 0:
        return np.zeros((nch, wlen))
    else:
        return np.roll(XCF_out, wlen // 2, axis=-1) / nwin

def find_noise_idx(data, noise_threshold=5, empty_tr=False):
    if not empty_tr:
        noisy_index = np.argmax(np.max(data, axis=1) > noise_threshold)
    else:
        noisy_index = np.argmax(np.linalg.norm(data, axis=1) < noise_threshold)
    return noisy_index

def impute_noisy_trace(data, noise_idx):
    if noise_idx + 1 == data.shape[0]:
        data[noise_idx] = data[noise_idx - 1]
    elif noise_idx == 0:
        data[noise_idx] = data[noise_idx + 1]
    else:
        data[noise_idx] = (data[noise_idx - 1] + data[noise_idx + 1])

def plot_xcorr(xcorr, t_axis, x_axis=None, ax=None, figsize=(8, 10),
               cmap='seismic', vmax_use_max=False,
               fig_dir=None,
               fig_name=None,
               fontsize=24, tickfont=20,
               x_lim=None,
               **plot_kwargs):
    if x_lim is None:
        x_lim = [-120, 120]
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    dt = t_axis[1] - t_axis[-1]
    nt = t_axis.size
    x_origin_index = np.abs(x_axis).argmin()
    xcorr /= np.amax(xcorr[x_origin_index])
    vmax = plot_kwargs.get("vmax", np.percentile(np.absolute(xcorr), 100)) if vmax_use_max else 1

    start_x = 0
    end_x = xcorr.shape[0]
    if x_axis is not None:
        start_x = -np.abs(x_axis[0])
        end_x = np.abs(x_axis[-1])

    if x_axis is not None:
        x_origin_idx = np.argmax(x_axis < 0)
        xcorr_to_plot = copy.deepcopy(xcorr)
        # xcorr_to_plot[x_origin_idx:] = np.flip(xcorr[x_origin_idx:], axis=-1)
    else:
        xcorr_to_plot = xcorr

    plt.imshow(xcorr_to_plot.T, aspect="auto", vmax=vmax, vmin=-vmax, cmap=cmap,
               extent=[start_x, end_x, t_axis[-1], t_axis[0]], interpolation='bicubic')
    # plt.ylim([t_lim, -t_lim])
    plt.xlabel("Distance along the fiber [m]", fontsize=fontsize)
    plt.ylabel("Time lag [s]", fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=tickfont)

    plt.xlim(x_lim)
    if fig_name and fig_dir:
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)
        print(f'{fig_path} has saved...')
    else:
        plt.show()


def normfunc(X):
    return np.asarray([X[:,i]/(abs(X[:,i]).max()+1e-20) for i in range(X.shape[1])]).T

class Dispersion:
    def __init__(self, data, dx, dt, freqs, vels, norm=False, compute_fv=True):
        self.data = data
        self.dx = dx
        self.dt = dt
        self.freqs = freqs
        self.vels = vels
        self.norm = norm
        if compute_fv:
            self._map_fv()

    def save_to_npz(self, fname, fdir='./'):
        np.savez(os.path.join(fdir, fname), freqs=self.freqs, vels=self.vels, fv_map=self.fv_map)

    @classmethod
    def get_dispersion_obj(cls, fname, fdir='./'):
        file = np.load(os.path.join(fdir, fname))
        obj = Dispersion(data=None, dx=None, dt=None, freqs=file['freqs'], vels=file['vels'], compute_fv=False)
        obj.fv_map = file['fv_map']
        return obj

    def _map_fv(self):
        self.fv_map = map_fv(self.data, self.dx, self.dt, freqs=self.freqs, vels=self.vels, norm=self.norm)

    def plot_image(self, fig_dir=None, fig_name=None, norm=False, **kwargs):
        norm = norm or self.norm
        print(fig_name)
        plot_fv_map(self.fv_map, self.freqs, self.vels, norm, fig_dir, fig_name, **kwargs)

    def __add__(self, other):
        sum_ = Dispersion(self.data, self.dx, self.dt, self.freqs, self.vels, compute_fv=False)
        sum_.fv_map = self.fv_map + other.fv_map
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other: float):
        div_ = copy.deepcopy(self)
        div_.fv_map /= other
        return div_


def map_fv_FD_slant_stack(data, dx, dt, freqs, vels, norm=True):
    data = data[6:25]
    if norm:
        data = data / np.linalg.norm(data, axis=-1, keepdims=True, ord=1)
    # data = data[::-1]
    # data = data[:200]
    # data = data[::-1]
    nt = data.shape[-1]
    nf = 2 ** (1 + math.ceil(math.log(nt, 2)))
    # data_fft = np.abs(np.fft.fft(data, axis=-1, n=nf))
    data_fft = np.fft.fft(data, axis=-1, n=nf)
    data_fft_freqs = np.fft.fftfreq(nf, d=dt)
    # freqs = data_fft_freqs[:data.shape[-1]//2]
    pout = np.zeros((freqs.size, vels.size), dtype=complex)

    for iv, v in enumerate(vels):
        for ix in range(data.shape[0]):
            x = dx * ix

            for fi, f in enumerate(freqs):
            # for fi, f in enumerate(freqs):
                arg = 2 * math.pi * f * x / v
                f_idx = np.abs(f - data_fft_freqs).argmin()
                pout[fi, iv] += data_fft[ix, f_idx] * (math.cos(arg) + 1j * math.sin(arg))
    pout = np.abs(pout)
    return pout.T


def map_fv(data, dx, dt, freqs, vels, norm=False):
    nscanv = np.size(vels)
    nscanf = np.size(freqs)

    if norm:
        # data = normfunc(data)
        # print('norm for disp...')
        data = data / np.linalg.norm(data, axis=-1, keepdims=True, ord=1)
    fk_res, fft_f, fft_k = fk(data, dx, dt)

    interp_fun = scipy.interpolate.interp2d(fft_k, fft_f, fk_res.T) # Transpose needed for interp2 definition

    ones_arr = np.ones(shape=(nscanv,))
    fv_map = np.zeros(shape=(len(freqs), len(vels)), dtype=np.float32)
    for ind, fr in enumerate(freqs):
        fv_map[ind, :] = np.squeeze(interp_fun(np.divide(ones_arr*fr, vels), fr))
    fv_map = scipy.signal.savgol_filter(fv_map, 25, 4, axis=0)

    return fv_map.T


def extract_ridge(freq, vel, fv_map, func_vel=None, sigma=25, vel_max = 400):
    # Velocity unit is m/s: vel, func_vel, sigma
    # Shape of fv_map: (Nvel, Nfreq)
    vel = vel[::-1]

    # No reference curve
    if func_vel is None:
        max_idx = np.abs(vel_max - vel).argmin()
        vel = vel[max_idx:]
        fv_map = fv_map[max_idx:]
        return vel[np.argmax(fv_map, axis=0)]

    # Extract ridgeline around the given curve
    else:
        # Reference velocity
        vel_ref = func_vel(freq)

        # Mask dispersion map
        vel_2d = np.tile(vel[::-1], (len(freq), 1)).T
        mask = (vel_2d > (vel_ref - sigma)) & (vel_2d < (vel_ref + sigma))
        mask_fv_map = np.ma.masked_array(fv_map, mask=~mask)

        # Dispersion curve
        return vel[np.argmax(mask_fv_map, axis=0)]

def map_fv_smooth(data, dx, dt, freqs, vels, norm=False):
    nscanv = np.size(vels)
    nscanf = np.size(freqs)

    if norm:
        # data = normfunc(data)
        data = data / np.linalg.norm(data, axis=-1, keepdims=True, ord=1)
    fk_res, fft_f, fft_k = fk(data, dx, dt)

    interp_fun = scipy.interpolate.interp2d(fft_k, fft_f, fk_res.T) # Transpose needed for interp2 definition

    ones_arr = np.ones(shape=(nscanv,))
    fv_map = np.zeros(shape=(len(freqs), len(vels)), dtype=np.float32)
    for ind, fr in enumerate(freqs):
        fv_map[ind, :] = np.squeeze(interp_fun(np.divide(ones_arr*fr, vels), fr))
    fv_map = scipy.signal.savgol_filter(fv_map, 13, 3, axis=0)

    return fv_map.T

def plot_fv_map(fv_map, freqs, vels, norm=True, fig_dir="Fig/", fig_name=None, ax=None, pclip=100, fontsize=24, tickfont=20, **kwargs):

    norm = True
    if norm:
        # row_sums = np.sqrt(np.sum(np.square(fv_map), axis=0))
        row_sums = np.amax(fv_map, axis=0)
        fv_map = fv_map / row_sums
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8,10)))

    pclip = 98
    vmax = np.percentile(np.abs(fv_map), pclip)
    vmin = np.percentile(np.abs(fv_map), 100-pclip)
#     print(vmax, vmin)

    ax.imshow(fv_map, aspect="auto",
              extent=[freqs[0], freqs[-1], vels[0], vels[-1]],
              cmap="jet",
              vmax=vmax,
              vmin=vmin)

    ax.grid()

    ax.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax.set_ylabel("Phase velocity [m/s]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfont)
    plt.tight_layout()
    if fig_name:
        fig_path = os.path.join(fig_dir, fig_name)
        print(f'saving {fig_path}...')
        plt.savefig(f"{fig_path}")
        plt.close()
    else:
        plt.show()


def bandpass_data_space(data, x_axis, flo, fhi):
    if flo == -1 and fhi == -1:
        return
    dx = x_axis[1] - x_axis[0]
    sampling_rate = int(1 / dx)

    fNy = 0.5 / dx
    order = 10
    # sos = signal.butter(order, [flo / fNy, fhi / fNy], analog=False, btype='band', output='sos')
    sos = signal.butter(order, [flo / fNy, fhi / fNy], analog=False, btype='band', output='sos')
    data[:] = signal.sosfiltfilt(sos, data, axis=0)
    # sos = signal.butter(order, [flo / fNy, fhi / fNy], analog=False, btype='band')
    # data[:] = signal.filtfilt(*sos, data, axis=0)
    # data[:] = signal.sosfilt(sos, data, axis=0)
    # print('new', time.time() - st)
    #
    # st = time.time()
    # for k, t_slice in enumerate(data.T):
    #     data[:, k] = bandpass(t_slice, freqmin=flo, freqmax=fhi, df=sampling_rate, corners=4, zerophase=True)
    # print('old', time.time() - st)

def upload_to_oas(oas_dir, local_path):
    flag = '-r' if os.path.isdir(local_path) else ''
    cmd = f"scp {flag} {local_path} {oas_dir}"
    print(cmd)
    os.system(cmd)
    if not flag:
        os.remove(local_path)
