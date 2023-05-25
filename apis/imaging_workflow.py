# Workflow

import os
import matplotlib.pyplot as plt
import time
import datetime
import argparse

import matplotlib

from modules.imaging_IO import ImagingIO
from apis.timeLapseImaging import TimeLapseImaging

DEFAULT_TRACKING_PARAM = {
    "detect":{
            "minprominence": 0.05,
            "minseparation": 20,
            "prominenceWindow": 600,
    }
}


class ImagingWorkflowOneDirectory:

    def __init__(self, directory, root, tracking_args=None, method='surface_wave', imaging_IO_dict={}):
        self.directory = directory
        self.root = root
        self.imagingIO = ImagingIO(directory, root, **imaging_IO_dict)
        self.time_interval = self.imagingIO.get_time_interval()
        self.tracking_args = tracking_args
        self.method = method

    def imaging(self, start_x, end_x, x0, wlen_sw=8, length_sw=300, spatial_ratio=0.75,
                n_min_save=30, temporal_spacing=None, num_to_stop=None, verbal=True,
                surface_wave_preprecessing_dict=None,
                imaging_kwargs=None):
        tracking_args = self.tracking_args if self.tracking_args else DEFAULT_TRACKING_PARAM

        avg_image = 0
        num_veh = 0
        self.avg_images_to_save = []
        n_win_save = int(n_min_save * 60 / self.time_interval)

        for k, (data, x_axis, t_axis) in enumerate(self.imagingIO):
            if num_to_stop and k >= num_to_stop:
                break
            tic = time.time()
            if verbal:
                print(f'working on window {k} / {len(self.imagingIO)}, method={self.method}')
            imagingObj = TimeLapseImaging(data, x_axis, t_axis, method=self.method, surface_wave_preprecessing_dict=surface_wave_preprecessing_dict)
            if verbal:
                print(f'detecting cars...')
            imagingObj.track_cars(start_x=start_x, end_x=end_x, tracking_args=tracking_args, show_plot=False)
            if verbal:
                print(f'selecting surface-wave windows...')
            imagingObj.select_surface_wave_windows(x0=x0, wlen_sw=wlen_sw, length_sw=length_sw, spatial_ratio=spatial_ratio, temporal_spacing=temporal_spacing)
            curt_veh_num = len(imagingObj.sw_selector)
            if curt_veh_num == 0:
                continue
            num_veh += curt_veh_num
            if verbal:
                print(f'Isolated cars: {curt_veh_num}; accumulated isolated cars: {num_veh}...')
                print(f'computing disp images...')
            imagingObj.get_images(**imaging_kwargs)
            if verbal:
                print(f'averaging...')
            avg_image += imagingObj.images.avg_image
            if k == 0 or (k + 1) % n_win_save == 0:
                result = {
                    "avg_image": avg_image,
                    "time": k * n_min_save,
                    "num_veh": num_veh
                }
                self.avg_images_to_save.append(result)
            toc = time.time()
            if verbal:
                print(f"time lapse: {toc - tic: .2f}s")

        self.avg_image = avg_image
        self.num_veh = num_veh

    def plot_avg_images(self, fname=None, figsize=(8, 8), norm=True, fig_dir='Fig/dispersion/', plot_xcorr_disp=False):
        fig, ax = plt.subplots(figsize=figsize)
        time_ = len(self.imagingIO) * self.time_interval

        ax.set_title(f"Time: {time_}m Number of Vehicles {self.num_veh}")

        if self.method == 'surface_wave':
            self.avg_image.plot_image(fname, norm=norm, ax=ax, fig_dir=fig_dir)
        else:
            self.avg_image.plot_image(fname, norm=norm, ax=ax, fig_dir=fig_dir, plot_disp=plot_xcorr_disp)


    def save_avg_disp_to_npz(self, *args, fdir=None, **kwargs):
        self.avg_image.save_to_npz(*args, fdir=fdir, **kwargs)

    def plot_intermediate_images(self, figsize=(10, 8), fig_dir='Fig/dispersion', x_lim=[-150, 150]):
        fig_folder = os.path.join(fig_dir, self.directory)
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)

        for k, result in enumerate(self.avg_images_to_save):
            time_ = k * 30
            n_cars = result['num_veh']
            name = f"time_{time_}m_nCars_{n_cars}"
            fname = f"vs_{name}.png"
            avg_image = result["avg_image"]
            avg_image.plot_image(fname, fig_dir=fig_folder, norm=True, x_lim=x_lim)
            disp_fname = f"disp_{name}.png"
            avg_image.compute_disp_image(end_x=0, start_x=-150)
            avg_image.plot_disp(fig_name=disp_fname, fig_dir=fig_folder)

def find_date_folders_for_date_range(start_date, end_date, root):
    date_folders = os.listdir(root)

    dir_list = []
    for folder in date_folders:
        dir_date = datetime.datetime.strptime(folder, "%Y%m%d")
        if start_date <= dir_date <= end_date:
            dir_list.append(folder)

    dir_list.sort()

    return dir_list

def dateStr_to_date(date_str):
    if isinstance(date_str, datetime.datetime):
        return date_str

    return datetime.datetime.strptime(date_str, "%Y-%m-%d")

def imaging_all_data(start_date, end_date, start_x=580, end_x=750, x0=675,
                     root='/net/brick5/scratch1/siyuan/time_lapse_windows/',
                     output_dir='Fig/dispersion/dates/',
                     fname_prefix='veh_avg_disp_'):

    start_date, end_date = dateStr_to_date(start_date), dateStr_to_date(end_date)

    dir_list = find_date_folders_for_date_range(start_date, end_date, root)
    if len(dir_list) == 0:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder in dir_list:
        print(f'working on {folder}...')
        workflow = ImagingWorkflowOneDirectory(folder)
        workflow.imaging(start_x, end_x, x0, verbal=False)
        fname = f'{fname_prefix}{folder}.png'
        fpath = os.path.join(output_dir, fname)
        workflow.plot_avg_images(fpath, norm=False)


class Imaging_for_multiple_date_range:

    def __init__(self, start_date, end_date, root='/net/brick5/scratch1/siyuan/time_lapse_windows/'):
        self.start_date = dateStr_to_date(start_date)
        self.end_date = dateStr_to_date(end_date)
        self.dir_list = find_date_folders_for_date_range(self.start_date, self.end_date, root)

    def imaging(self, start_x=580, end_x=750, x0=675,
                wlen_sw=12,
        output_npz_dir='/net/brick5/scratch1/siyuan/time_lapse_disp_results',
                verbal=False,
                method='surface_wave'):
        if method == 'surface_wave':
            fname_prefix = 'veh_avg_disp_'
            output_fig_dir = 'Fig/dispersion/dates/'
        else:
            fname_prefix = 'veh_avg_xcorr_'
            output_fig_dir = 'Fig/vs/dates/'

        if len(self.dir_list) == 0:
            return

        self.workflows = {}


        for folder in self.dir_list:
            print(f'working on {folder}...')
            fname = f'{fname_prefix}{folder}.png'
            fname_norm = f'{fname_prefix}{folder}_normed.png'
            fname_npz = f'{fname_prefix}{folder}.npz'
            fpath = os.path.join(output_fig_dir, fname)
            fpath_norm = os.path.join(output_fig_dir, fname_norm)
            fpath_npz = os.path.join(output_npz_dir, fname_npz)

            if os.path.exists(fpath) and os.path.exists(fpath_norm) and os.path.exists(fpath_npz):
                print(f'{fpath} exists, continued...')
                continue

            workflow = ImagingWorkflowOneDirectory(folder, method=method)
            workflow.imaging(start_x, end_x, x0, verbal=verbal, wlen_sw=wlen_sw)
            workflow.plot_avg_images(fname, norm=False, fig_dir=output_fig_dir)
            workflow.plot_avg_images(fname_norm, norm=True, fig_dir=output_fig_dir)
            if method == 'xcorr':
                # from virtual shot gather image to disp image
                workflow.avg_image.compute_disp_image()
                workflow.plot_avg_images()
            workflow.save_avg_disp_to_npz(fname=fname_npz, fdir=output_npz_dir)

            self.workflows[folder] = workflow


if __name__ == "__main__":
    # matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='argparse for imaging DAS data for a given date range')

    parser.add_argument('--start_date', type=str, default='2022-12-02',
                        help='pacific time in the format of %Y-%m-%d')

    parser.add_argument('--end_date', type=str, default='2022-12-02',
                        help='pacific time in the format of %Y-%m-%d')

    parser.add_argument('--verbal', action="store_true")

    args = parser.parse_args()

    imaging = Imaging_for_multiple_date_range(args.start_date, args.end_date)

    imaging.imaging(verbal=args.verbal)
