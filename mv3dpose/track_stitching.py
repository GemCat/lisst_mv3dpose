import sys
import os
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
    conf = Config(dataset_dir, vis=True)

    tracks = []
    print()
    nr_tracks = 0
    for path in os.scandir(conf.output_dir):
        if path.is_file():
            nr_tracks += 1
            
    for i in range(nr_tracks):
        filename = join(conf.output_dir, f'track{i}.json')
        track = Track.from_file(filename)
        tracks.append(track)


    print(f"Tracks read: {len(tracks)}")

    # print(tracks[0].poses[0])
    # print(np.asarray(tracks[0].poses[0]))
    # T R A C K S   S T I T C H I N G 
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

    def merge_tracks(trackA, trackB):
        new_poses = trackA.poses + trackB.poses
        frames = trackA.frames + trackB.frames

        last_seen_delay = 99 #trackB.last_seen_delay
        z_axis = trackB.z_axis
        frame0 = frames.pop(0)
        pose0 = new_poses.pop(0)

        new_track = Track(frame0, pose0, last_seen_delay, z_axis)
        for t, pose in zip(frames, new_poses):
           new_track.add_pose(t, pose)
        return new_track


    stitched_tracks = []
    for i, track in enumerate(tracks):
        to_stitch = False
        for id1 in stitch:
            if stitch[id1]['track2'] == i:
                stitched_tracks.append(merge_tracks(tracks[id1], tracks[i]))
                to_stitch = True
        if not to_stitch:
            stitched_tracks.append(tracks[i])
    print(len(stitched_tracks))
    print(len(tracks)) 
    # delete output dir
    # if isdir(self.output_dir) and not vis:
    #         print('\n[deleting existing output directory...]')
    #         shutil.rmtree(self.output_dir)