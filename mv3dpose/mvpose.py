import sys
sys.path.insert(0, '../')
from mv3dpose.tracking import tracking, Track
from mv3dpose.config import Config
from mv3dpose.load_functions import load_keypoints, load_cameras
from os.path import join
from tqdm import tqdm


if __name__ == '__main__':
    
    # the first argument is the path to dataset dir  
    dataset_dir = sys.argv[1]
    conf = Config(dataset_dir)

    poses_per_frame = load_keypoints(conf)
    calib = load_cameras(conf)

    # T R A C K I N G
    tracks = tracking(calib, poses_per_frame, conf)
    
    # S M O O T H I N G
    if conf.do_smoothing:
        print('\n[smoothing]')
        tracks_ = []
        for track in tqdm(tracks):
            track = Track.smoothing(track,
                                    sigma=conf.smoothing_sigma,
                                    interpolation_range=conf.smoothing_interpolation_range)
            tracks_.append(track)
        tracks = tracks_

    # S E R I A L I Z E
    print('\n[serialize 3d tracks]')
    for tid, track in tqdm(enumerate(tracks)):
        fname = join(conf.output_dir, 'track' + str(tid) + '.json')
        track.to_file(fname)
