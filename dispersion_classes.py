import copy

import numpy as np

from data_classes import SurfaceWaveWindow
from utils import Dispersion

class SurfaceWaveDispersion:

    def __init__(self, window: SurfaceWaveWindow, freqs=np.arange(0.8, 25, 0.1), vels=np.arange(200, 1200),
                 method="naive", norm=True, **method_kwargs):
        self.window = window
        self.freqs = freqs
        self.vels = vels
        self.method = method
        self.norm = norm
        if method == "naive":
            self._naive_disp(**method_kwargs)
        else:
            self._smart_disp(**method_kwargs)

    # def _naive_disp(self, start_x=80, dist=150):
    def _naive_disp(self, start_x, end_x):
        dist = end_x - start_x

        window = self.window
        dx = window.x_axis[1] - window.x_axis[0]
        start_x_idx = np.argmax(window.x_axis >= start_x)
        nx = int(dist / dx)
        self.disp = Dispersion(window.data[start_x_idx:start_x_idx + nx], dx, window.t_axis[1] - window.t_axis[0],
                               freqs=self.freqs, vels=self.vels, norm=self.norm)

    def _smart_disp(self, mute_along_time=True, time_alpha=0.3, mute_along_traj=True):
        window_mute = copy.deepcopy(self.window)
        if mute_along_time and not getattr(window_mute, 'muted_along_time', False):
            window_mute.mute_along_time(alpha=time_alpha)

        if mute_along_traj and not getattr(window_mute, 'muted_along_traj', False):
            window_mute.mute_along_traj()
        dx = window_mute.x_axis[1] - window_mute.x_axis[0]
        self.disp = Dispersion(window_mute.data, dx, window_mute.t_axis[1] - window_mute.t_axis[0],
                               freqs=self.freqs, vels=self.vels, norm=self.norm)

    def plot_image(self, fig_name=None, fig_dir="Fig/dispersion/", norm=False, **kwargs):
        self.disp.plot_image(fig_dir, fig_name, norm=norm, **kwargs)

    def save_to_npz(self, *args, **kwargs):
        self.disp.save_to_npz(*args, **kwargs)

    def __add__(self, other):
        sum_ = copy.deepcopy(self)
        sum_.disp = self.disp + other.disp
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __truediv__(self, other: float):
        new_obj = copy.deepcopy(self)
        new_obj.disp = self.disp / other
        return new_obj
