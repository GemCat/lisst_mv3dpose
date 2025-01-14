import sys
sys.path.insert(0, '../')
from mv3dpose.tracking import tracking, Track
from mv3dpose.config import Config
from mv3dpose.load_functions import load_keypoints, load_cameras
from os.path import join
from tqdm import tqdm

from mv3dpose.hypothesis import get_distance3d
import numpy as np


if __name__ == '__main__':
    
    # the first argument is the path to dataset dir  
    dataset_dir = sys.argv[1]
    conf = Config(dataset_dir)

    poses_per_frame = load_keypoints(conf)
    calib = load_cameras(conf)

    # T R A C K I N G
    tracks = tracking(calib, poses_per_frame, conf, distance_threshold=200)
    
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

    dist_threshold = 250
    counter = 0
    stitch = {}
    for i1, track1 in enumerate(tracks):
        for i2, track2 in enumerate(tracks):
            frame_dist = track2.frames[0] - track1.frames[-1]
            if (track1 != track2 and
                frame_dist > 0 and 
                frame_dist < 2*conf.min_track_length):
                    pose_dist = np.mean(get_distance3d(track1.poses[-1], track2.poses[0]))*conf.scale_to_mm
                    if (pose_dist < dist_threshold):
                        if i1 in stitch:
                            if stitch[i1]['frame_dist'] < frame_dist:
                                inner_dict = {}
                                inner_dict['track2']     = i2 
                                inner_dict['pose_dist']  = pose_dist
                                inner_dict['frame_dist'] = frame_dist
                                stitch[i1] = inner_dict

                        else:
                            inner_dict = {}
                            inner_dict['track2']     = i2 
                            inner_dict['pose_dist']  = pose_dist
                            inner_dict['frame_dist'] = frame_dist
                            stitch[i1] = inner_dict

    print("++++++++++++++++")
    print(stitch)
    # print(counter)
    print("++++++++++++++++")
    # def merge_tracks(trackA, trackB):
    #     new_poses = trackA.poses + trackB.poses
    #     frames = trackA.frames + trackB.frames

    #     last_seen_delay = 99 #trackB.last_seen_delay
    #     z_axis = trackB.z_axis
    #     frame0 = frames.pop(0)
    #     pose0 = new_poses.pop(0)

    #     new_track = Track(frame0, pose0, last_seen_delay, z_axis)
    #     for t, pose in zip(frames, new_poses):
    #        new_track.add_pose(t, pose)
    #     return new_track


    # stitched_tracks = []
    # for i, track in enumerate(tracks):
    #     to_stitch = False
    #     for id1 in stitch:
    #         if stitch[id1]['track2'] == i:
    #             stitched_tracks.append(merge_tracks(tracks[id1], tracks[i]))
    #             to_stitch = True
    #     if not to_stitch:
    #         stitched_tracks.append(tracks[i])
    # print(len(stitched_tracks))
    # print(len(tracks))      

    # S E R I A L I Z E
    print('\n[serialize 3d tracks]')
    for tid, track in tqdm(enumerate(tracks)):
        fname = join(conf.output_dir, 'track' + str(tid) + '.json')
        track.to_file(fname)
