import sys
sys.path.insert(0, '../')
from os.path import isdir, join
from os import makedirs
import json
import shutil

class Config:
    def __init__(self, dataset_dir, vis = False):
        self.dataset_dir = dataset_dir
        dataset_json = join(dataset_dir, 'dataset.json')
        settings = json.load(open(dataset_json))

        self.output_dir = join(dataset_dir, settings['output_dir'])  
        self.json_poses_prefix = settings['json_poses_prefix']
        self.cam_dir     = join(dataset_dir, settings['camera_calib_dir'])
        self.kyp_dir     = join(dataset_dir, settings['keypoints_dir'])
        self.scale_to_mm = settings['scale_to_mm']
        self.n_cameras   = settings['n_cameras']

        assert isdir(self.cam_dir)
        assert isdir(self.kyp_dir)

        if isdir(self.output_dir) and not vis:
            print('\n[deleting existing output directory...]')
            shutil.rmtree(self.output_dir)
            
        if not vis:
            makedirs(self.output_dir)

        # ~~~~~~~~~~~~~~~~~~~~~~

        self.smoothing_interpolation_range = settings['smoothing_interpolation_range']
        self.max_distance_between_tracks = settings['max_distance_between_tracks']
        self.epi_threshold    = settings['epi_threshold']
        self.min_track_length = settings['min_track_length']
        self.merge_distance   = settings['merge_distance']
        self.last_seen_delay  = settings['last_seen_delay']
        self.smoothing_sigma  = settings['smoothing_sigma']
        self.do_smoothing     = settings['do_smoothing']
        self.valid_frames     = list(range(settings['valid_frames']))

        print("\n#frames", len(self.valid_frames))

        if vis:
            self.trk_dir = self.output_dir
            assert isdir(self.trk_dir), "the tracks must be extracted!"
            self.img_dir = join(dataset_dir, 'img')

            self.img_file_type = settings["img_file_type"]

            self.vis_dir = join(dataset_dir, settings['vis_dir'])
            if isdir(self.vis_dir):
                shutil.rmtree(self.vis_dir)
            makedirs(self.vis_dir)

            self.vis_frames = settings['vis_frames']
            self.plot3d = settings['plot3d']