import copy

from dispersion_classes import SurfaceWaveDispersion
from virtual_shot_gather import VirtualShotGather


class ImagesFromWindows:
    def __init__(self, windows, image_cls):
        """

        :param windows: List of SurfaceWaveWindow obj
        """
        self.windows = windows
        self.image_cls = image_cls

    def get_images(self, norm=False, mute_offset=300, mute=True, **imaging_kwargs):
        self.images = []

        for k, window in enumerate(self.windows):
            if mute and not window.muted_along_traj:
                window = copy.deepcopy(window)
                window.mute_along_traj(offset=mute_offset)
            image = self.image_cls(window, norm=norm, **imaging_kwargs)
            self.images.append(image)

        self.avg_image = sum(self.images)
        self.avg_image = self.avg_image / len(self.images)


    def save_images(self, fig_folder, file_prefix):

        for k, image in enumerate(self.images):
            fname = f"{file_prefix}{k}.png"
            image.plot_image(fname, norm=True, fig_folder=fig_folder)

        fname = f"{file_prefix}_avg.png"
        self.avg_image.plot_image(fname, norm=True, fig_folder=fig_folder)


class DispersionImagesFromWindows(ImagesFromWindows):

    def __init__(self, windows, image_cls=SurfaceWaveDispersion):
        super().__init__(windows, image_cls)

    def save_images(self, fig_folder, file_prefix='veh_disp'):
        super(DispersionImagesFromWindows, self).save_images(fig_folder, file_prefix)


class VirtualShotGathersFromWindows(ImagesFromWindows):
    def __init__(self, windows, image_cls=VirtualShotGather):
        """

        :param windows: List of SurfaceWaveWindow obj
        """
        super().__init__(windows, image_cls)

    def get_images(self, norm=False, mute_offset=300, mute=False, **imaging_kwargs):
        super().get_images(norm=False, mute_offset=300, mute=False, **imaging_kwargs)

    def save_images(self, fig_folder, file_prefix='veh_vshot'):
        super(VirtualShotGathersFromWindows, self).save_images(fig_folder, file_prefix)
