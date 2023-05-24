import sys
import os
sys.path.insert(0, '../')
from mv3dpose.tracking import Track
from mv3dpose.config import Config
from os.path import join

from mv3dpose.hypothesis import get_distance3d
import numpy as np


def merge_tracks(curr_track, next_track, frame):
    new_poses = []
    frames = []

    # if tracks don't overlap, combine everything
    if curr_track.frames[-1] < frame:
            new_poses = curr_track.poses + next_track.poses
            frames = curr_track.frames + next_track.frames
    
    # if tracks do overlap, add only the frames and poses from the current track
    # up to the first frame and pose of the next track. Add everything from 
    # the next track
    else:
        idx = 0
        while (curr_track.frames[idx] < frame):
            new_poses.append(curr_track.poses[idx])
            frames.append(curr_track.frames[idx])
            idx += 1

        new_poses += next_track.poses
        frames += next_track.frames

    last_seen_delay = 99 
    z_axis = curr_track.z_axis
    frame0 = frames.pop(0)
    pose0 = new_poses.pop(0)

    new_track = Track(frame0, pose0, last_seen_delay, z_axis)
    for t, pose in zip(frames, new_poses):
        new_track.add_pose(t, pose)
    return new_track


if __name__ == '__main__':
    
    # the first argument is the path to dataset dir  
    # dataset_dir = sys.argv[1]
    dataset_dir = '/content/drive/MyDrive/lisst_mv3dpose/dataset'
    conf = Config(dataset_dir, vis=True)

    tracks = []
    nr_tracks = 0
    for path in os.scandir(conf.output_dir):
        if path.is_file():
            nr_tracks += 1
            
    for i in range(nr_tracks):
        filename = join(conf.output_dir, f'track{i}.json')
        track = Track.from_file(filename)
        tracks.append(track)

    print(f"Tracks read: {len(tracks)}")

    # T R A C K S   S T I T C H I N G 
    dist_threshold = 100
    max_frame_dist = 2*conf.min_track_length

    new_tracks = {}
    nid = 0
    discard = False
    merged = False

    for tid, track in enumerate(tracks):

        # Establish each track that has frame 0 as a new track
        if track.frames[0] == 0:
            new_tracks[nid] = track
            nid += 1
            continue
        
        next_frame = track.frames[0]
        next_pose = track.poses[0]

        # Try to match the next track to an already existing new track
        for id in range(nid):
            new_track = new_tracks[id]
            if next_frame < new_track.frames[-1]:
                pose_idx = new_track.frames.index(next_frame)
                new_pose = new_track.poses[pose_idx]
            else:
                frame_dist = next_frame - new_track.frames[-1]
                if frame_dist > max_frame_dist:
                    continue
                new_pose = new_track.poses[-1]

            pose_dist = np.mean(get_distance3d(new_pose, next_pose))

            if pose_dist < dist_threshold:
                # discard track if its pose matches but its frames are already 
                # encompassed by the current new track
                if track.frames[-1] < new_track.frames[-1]:  
                    discard = True
                    break

                merged_track = merge_tracks(new_track, track, next_frame)
                new_tracks[id] = merged_track
                merged = True
                break


        if discard or merged:
                merged = False
                discard = False
                continue
        
        # Add track as a new track if it isn't merged or discarded 
        new_tracks[nid] = track
        nid += 1
            
    print(f"Tracks after stitching: {len(new_tracks.keys())}")
    stitch_dir = join(conf.dataset_dir, "stitched")
    print('\n[serialize 3d tracks]')

    for tid, track in new_tracks.items():
        fname = join(stitch_dir, 'track' + str(tid) + '.json')
        track.to_file(fname)