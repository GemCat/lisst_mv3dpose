import sys
sys.path.insert(0, '../')
from mv3dpose.data.openpose import OpenPoseKeypoints, MultiOpenPoseKeypoints
from mv3dpose.tracking import tracking, Track
from mv3dpose.config import Config
from mv3dpose.load_functions import load_keypoints, load_cameras
import mv3dpose.geometry.camera as camera
from os.path import isdir, join, isfile
from os import makedirs, listdir
from tqdm import tqdm
from time import time
import json
import numpy as np


if __name__ == '__main__':
    
    # the first argument is the path to dataset dir  
    dataset_dir = sys.argv[1]
    conf = Config(dataset_dir)

    print("\n#frames", len(conf.valid_frames))

    pe    = load_keypoints(conf)
    calib = load_cameras(conf)

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # L O A D  2 D  P O S E S
    # ~~~~~~~~~~~~~~~~~~~~~~~
    print('\n[load 2d poses]')
    poses_per_frame = []
    for frame in tqdm(conf.valid_frames):
        predictions = pe.predict(frame)
        poses_per_frame.append(predictions)

    # ~~~~~~~~~~~~~~~
    # T R A C K I N G
    # ~~~~~~~~~~~~~~~
    print('\n[tracking]')
    _start = time()
    tracks = tracking(calib, poses_per_frame,
                    epi_threshold=conf.epi_threshold,
                    scale_to_mm=conf.scale_to_mm,
                    max_distance_between_tracks=conf.max_distance_between_tracks,
                    actual_frames=conf.valid_frames,
                    min_track_length=conf.min_track_length,
                    merge_distance=conf.merge_distance,
                    last_seen_delay=conf.last_seen_delay)
    _end = time()
    print('\telapsed', _end - _start)
    print("\n\t# detected tracks:", len(tracks))

    # ~~~~~~~~~~~~~~~
    # S M O O T H I N G
    # ~~~~~~~~~~~~~~~
    if conf.do_smoothing:
        print('\n[smoothing]')
        tracks_ = []
        for track in tqdm(tracks):
            track = Track.smoothing(track,
                                    sigma=conf.smoothing_sigma,
                                    interpolation_range=conf.smoothing_interpolation_range)
            tracks_.append(track)
        tracks = tracks_

    # ~~~~~~~~~~~~~~~~~
    # S E R I A L I Z E
    # ~~~~~~~~~~~~~~~~~
    print('\n[serialize 3d tracks]')
    for tid, track in tqdm(enumerate(tracks)):
        fname = join(conf.output_dir, 'track' + str(tid) + '.json')
        track.to_file(fname)
