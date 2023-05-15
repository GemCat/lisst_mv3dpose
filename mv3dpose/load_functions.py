from mv3dpose.config import Config
import sys
sys.path.insert(0, '../')
from mv3dpose.data.openpose import OpenPoseKeypoints, MultiOpenPoseKeypoints
from mv3dpose.config import Config
import mv3dpose.geometry.camera as camera
from os.path import isdir, join, isfile
from tqdm import tqdm

# ~~~~~~~~~~~~~~~~~
# K E Y P O I N T S
# ~~~~~~~~~~~~~~~~~
def load_keypoints(config: Config):
    print('\n[load keypoints]')
    keypoints = []
    for cid in tqdm(range(config.n_cameras)):
        # directories with json files wrt each camera  
        loc = join(config.kyp_dir, f'{config.json_poses_prefix}%02d_json' % cid)
        assert isdir(loc), loc
        file_prefix = f'{config.json_poses_prefix}%02d' % cid
        pe = OpenPoseKeypoints(f'{file_prefix}_%012d', loc)
        keypoints.append(pe)

    pe = MultiOpenPoseKeypoints(keypoints)
    return pe


# ~~~~~~~~~~~~~
# C A M E R A S
# ~~~~~~~~~~~~~
def load_cameras(config: Config):
    print('\n[load cameras]')
    calib = []
    for cid in tqdm(range(config.n_cameras)):
        cam_fname = join(config.cam_dir, 'camera%02d_calibration.json' % cid)
        assert isfile(cam_fname), cam_fname
        cam = camera.Camera.load_from_file(cam_fname)
        calib.append(cam)
    return calib