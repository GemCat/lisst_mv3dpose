import sys
import os
sys.path.insert(0, '../')
from mv3dpose.tracking import Track
from mv3dpose.config import Config
from os.path import join

import numpy as np
import numpy.linalg as la


def get_distance3d(person1, person2):
    J = len(person1)
    assert len(person2) == J
    result = []
    max_diff = 0
    for jid in range(J):
        if person1[jid] is None or person2[jid] is None:
            continue
        if (isinstance(person1[jid], list)):
            person1_pose = np.asarray(person1[jid])
            person2_pose = np.asarray(person2[jid])
        else:
            person1_pose = person1[jid]
            person2_pose = person2[jid]
        d = la.norm(person1_pose - person2_pose)
        # print("jid: " + str(jid) + " distance: " + str(d))
        if d > max_diff:
            max_diff = d
        result.append(d)

    if result == []:
        return np.array([100])

    result.remove(max_diff)
    return np.array(result)

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

    last_seen_delay = 120
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
    max_frame_dist = 60

    new_tracks = {}
    nid = 0
    for tid, track in enumerate(tracks):

        print("Track: " + str(tid))
        # Establish each track that has frame 0 as a new track
        if track.frames[0] == 0:
            new_tracks[nid] = track
            nid += 1
            continue

        next_frame = track.frames[0]
        next_pose = track.poses[0]
        min_dist = np.inf
        closest_tid = None
        # Try to match the next track to an already existing new track
        for id in range(nid):
            new_track = new_tracks[id]
            if next_frame < new_track.frames[-1]:
                pose_idx = new_track.frames.index(next_frame)
                new_pose = new_track.poses[pose_idx]
                dist_threshold = 55
            else:
                frame_dist = next_frame - new_track.frames[-1]
                if frame_dist > max_frame_dist:
                    continue
                new_pose = new_track.poses[-1]
                dist_threshold = 100

            pose_dist = np.mean(get_distance3d(new_pose, next_pose))
            if pose_dist < min_dist:
                min_dist = pose_dist
                closest_tid = id
                curr_thresh = dist_threshold
            print("distance between poses for new track " + str(id) + ": " + str(pose_dist))

        if min_dist < curr_thresh:
            new_track = new_tracks[closest_tid]
            # discard track if its pose matches but its frames are already
            # encompassed by the current new track
            if track.frames[-1] < new_track.frames[-1]:
                print("discarded, closest to track " + str(closest_tid))
                continue
            else:
                merged_track = merge_tracks(new_track, track, next_frame)
                new_tracks[closest_tid] = merged_track
                print("Merged tracks: " + str(closest_tid) + " and " + str(tid))
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

    final = []
    for tid, track in new_tracks.items():
        track = new_tracks[tid]
        track_length = track.frames[-1] - track.frames[0]
        if track_length > 60:
            final.append(track)

    print("Tracks after removing less than 2 seconds: " + str(len(final)))
    for tid, track in enumerate(final):
        fname = join(stitch_dir, 'final_track' + str(tid) + '.json')
        track.to_file(fname)