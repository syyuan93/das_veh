from imaging_workflow import ImagingWorkflowOneDirectory
from virtual_shot_gather import VirtualShotGather
import os
from utils import Dispersion

tracking_args = {
    "detect":{
            "minprominence": 0.2,
            "minseparation": 40,
            "prominenceWindow": 600,
    }
}

def check_if_res_exist(res_dir, res_name):
    res_path = os.path.join(res_dir, res_name)
    return os.path.exists(res_path)

def test_workflow_temporal_spacings(date, temporal_spacings, pivot,
                 res_root = '/net/brick6/data5/syyuan/time_lapse_imaging/tests/temporal_spacings',
                 vs_fig_folder_name='Fig/virtual_shot_gather',
                 vs_res_folder_name='Res/virtual_shot_gather',
                 disp_fig_folder_name='Fig/dispersion_images/',
                 imaging_offset=200,
                 tracking_offset=180,
                 num_to_stop=None):

    for temporal_spacing in temporal_spacings:
        workflow = ImagingWorkflowOneDirectory(date, method='xcorr', tracking_args=tracking_args)
        workflow.imaging(start_x=pivot - tracking_offset, end_x=pivot + tracking_offset, x0=pivot,
                         wlen_sw=30, length_sw=500, spatial_ratio=0.5,
                         num_to_stop=num_to_stop,
                         temporal_spacing=temporal_spacing,
                         imaging_kwargs={"pivot": pivot, "start_x": pivot - imaging_offset,
                                         "end_x": pivot + imaging_offset, "wlen": 2,
                                         "include_other_side": True})

        avg_image = workflow.avg_image
        vs_name = f"{date}_vs_pivot_{pivot}_temporalSpacing_{temporal_spacing}_vehNum_{workflow.num_veh}"
        vs_fig_dir = os.path.join(res_root, date, vs_fig_folder_name)
        if not os.path.exists(vs_fig_dir):
            os.makedirs(vs_fig_dir)

        avg_image.plot_image(fig_name=vs_name, fig_dir=vs_fig_dir)

        vs_res_dir = os.path.join(res_root, date, vs_res_folder_name)
        if not os.path.exists(vs_res_dir):
            os.makedirs(vs_res_dir)
        avg_image.save_to_npz(fname=vs_name,
                              fdir=vs_res_dir,
                              num_veh=workflow.num_veh)

        disp_res_dir = os.path.join(res_root, date, disp_fig_folder_name)
        if not os.path.exists(disp_res_dir):
            os.makedirs(disp_res_dir)
        avg_image.compute_disp_image(end_x=0, start_x=150)

        disp_name = f'{date}_disp_pivot_{pivot}_temporalSpacing_{temporal_spacing}_vehNum_{workflow.num_veh}'
        avg_image.plot_disp(fig_name=disp_name, fig_dir=disp_res_dir)


def test_workflow_include_whole_data(date, to_combine, temporal_spacing=12, pivot=700,
                 res_root = '/net/brick6/data5/syyuan/time_lapse_imaging/tests/include_whole_data',
                 vs_fig_folder_name='Fig/virtual_shot_gather',
                 vs_res_folder_name='Res/virtual_shot_gather',
                 disp_fig_folder_name='Fig/dispersion_images/',
                 imaging_offset=200,
                 tracking_offset=180,
                 num_to_stop=None,
                 method='xcorr',
                 imaging_IO_dict={},
                 length_sw=500,
                 surface_wave_preprecessing_dict=None,
                 default_tracking_args=None,
                 recompute=False,
                                     ):

    include_other_side = to_combine

    if not default_tracking_args:
        default_tracking_args = tracking_args
    vs_name = f"{date}_vs_pivot_{pivot}"
    if include_other_side:
        vs_name += "_combined"

    res_exists = False
    vs_res_dir = os.path.join(res_root, date, vs_res_folder_name)
    if not os.path.exists(vs_res_dir):
        os.makedirs(vs_res_dir)
    else:
        res_exists = check_if_res_exist(vs_res_dir, vs_name + ".npz")

    if not res_exists or recompute:
        try:
            workflow = ImagingWorkflowOneDirectory(date,
                                                   method=method,
                                                   tracking_args=default_tracking_args,
                                                   imaging_IO_dict=imaging_IO_dict)
        except Exception as e:
            print(e)
            return
        workflow.imaging(start_x=pivot - tracking_offset, end_x=pivot + tracking_offset, x0=pivot,
                     wlen_sw=25, length_sw=length_sw, spatial_ratio=0.5,
                     num_to_stop=num_to_stop,
                     temporal_spacing=temporal_spacing,
                     surface_wave_preprecessing_dict=surface_wave_preprecessing_dict,
                     imaging_kwargs={"pivot": pivot, "start_x": pivot - imaging_offset, "end_x": pivot + imaging_offset, "wlen": 2,
                                    "include_other_side": include_other_side})

        avg_image = workflow.avg_image
    else:
        print('results already exists...')
        avg_image = VirtualShotGather.get_VirtualShotGather_obj(vs_res_dir, vs_name + ".npz")

    vs_fig_dir = os.path.join(res_root, date, vs_fig_folder_name)
    if not os.path.exists(vs_fig_dir):
        os.makedirs(vs_fig_dir, exist_ok=True)

    if type(avg_image) == int:
        print('avg_image is int..., continued...')
        return
    avg_image.plot_image(fig_name=vs_name, fig_dir=vs_fig_dir)

    if not res_exists:
        avg_image.save_to_npz(fname=vs_name,
                              fdir=vs_res_dir)

    disp_res_dir = os.path.join(res_root, date, disp_fig_folder_name)
    if not os.path.exists(disp_res_dir):
        os.makedirs(disp_res_dir)
    avg_image.compute_disp_image(end_x=0, start_x=-150)

    disp_name = f'{date}_disp_pivot_{pivot}'
    if include_other_side:
        disp_name += "_combined"
    avg_image.plot_disp(fig_name=disp_name, fig_dir=disp_res_dir)



def test_workflow_method(date, to_combine=True, temporal_spacing=12, pivot=700,
                 res_root = '/net/brick6/data5/syyuan/time_lapse_imaging/tests/method/',
                 vs_fig_folder_name='Fig/virtual_shot_gather',
                 vs_res_folder_name='Res/virtual_shot_gather',
                 disp_fig_folder_name='Fig/dispersion_images/',
                 imaging_offset=200,
                 tracking_offset=180,
                 num_to_stop=None,
                 method='xcorr'
                                     ):

    include_other_side = to_combine

    vs_name = f"{date}_vs_pivot_{pivot}_method_{method}"

    res_exists = False
    vs_res_dir = os.path.join(res_root, date, vs_res_folder_name)
    if not os.path.exists(vs_res_dir):
        os.makedirs(vs_res_dir)
    else:
        res_exists = check_if_res_exist(vs_res_dir, vs_name + ".npz")

    if method == 'xcorr':
        imaging_kwargs = {"pivot": pivot, "start_x": pivot - imaging_offset, "end_x": pivot + imaging_offset, "wlen": 2,
                                    "include_other_side": include_other_side}
    else:
        imaging_kwargs = {"start_x": pivot - 150, "end_x": pivot}

    if not res_exists:
        workflow = ImagingWorkflowOneDirectory(date, method=method, tracking_args=tracking_args)
        workflow.imaging(start_x=pivot - tracking_offset, end_x=pivot + tracking_offset, x0=pivot,
                     wlen_sw=25, length_sw=500, spatial_ratio=0.5,
                     num_to_stop=num_to_stop,
                     temporal_spacing=temporal_spacing,
                     imaging_kwargs=imaging_kwargs)

        avg_image = workflow.avg_image
    else:
        print('results already exists...')
        if method == 'xcorr':
            avg_image = VirtualShotGather.get_VirtualShotGather_obj(vs_res_dir, vs_name + ".npz")
        else:
            avg_image = Dispersion.get_dispersion_obj(vs_name + ".npz", vs_res_dir)

    vs_fig_dir = os.path.join(res_root, date, vs_fig_folder_name)
    if not os.path.exists(vs_fig_dir):
        os.makedirs(vs_fig_dir)

    avg_image.plot_image(fig_name=vs_name, fig_dir=vs_fig_dir)

    if not res_exists:
        avg_image.save_to_npz(fname=vs_name,
                              fdir=vs_res_dir)

    if method == 'xcorr':
        disp_res_dir = os.path.join(res_root, date, disp_fig_folder_name)
        if not os.path.exists(disp_res_dir):
            os.makedirs(disp_res_dir)
        avg_image.compute_disp_image(end_x=0, start_x=-150)

        disp_name = f'{date}_disp_pivot_{pivot}_method_{method}'
        avg_image.plot_disp(fig_name=disp_name, fig_dir=disp_res_dir)
