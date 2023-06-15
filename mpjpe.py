import numpy as np
import json
import os

# gt_dir = "/home/piotr/panoptic-toolbox/scripts/161029_build1/hdPose3d_stage1_coco19"
# track_dir = "/home/piotr/Downloads/build_tracks"
# frame_offset = 235

# gt_dir = "/home/piotr/panoptic-toolbox/scripts/160224_haggling1/hdPose3d_stage1_coco19/hd"
# track_dir = "/home/piotr/Downloads/haggle_tracks"
# frame_offset = 253

gt_dir = "/home/piotr/panoptic-toolbox/scripts/160906_pizza1/hdPose3d_stage1_coco19"
track_dir = "/home/piotr/Downloads/pizza_tracks"
frame_offset = 0



def get_frame_poses(frame_json):
    if not os.path.exists(frame_json):
        return None
    with open(frame_json, "r") as file:
        data = json.load(file)
        return data["bodies"]
    
joint_to_ours_map = [1, 0, None, 5, 6, 7, 11, 12, 13, 2, 3, 4, 8, 9, 10, 15, 17, 14, 17]

def joint19_to_ours(joint19):
    ours = [None for i in range(30)]
    for i in range(19):
        if joint_to_ours_map[i] is not None:
            ours[joint_to_ours_map[i]] = [joint19[4*i], joint19[4*i+1], joint19[4*i+2]]
    return ours

def calculate_mpjpe(pred, gt):
    gt = joint19_to_ours(gt["joints19"])
    mpjpe = 0
    n = 0
    for i in range(len(pred)):
        if pred[i] is not None and gt[i] is not None:
            mpjpe += np.sqrt(pow(pred[i][0]-gt[i][0], 2)+pow(pred[i][1]-gt[i][1], 2)+pow(pred[i][2]-gt[i][2], 2))
            n += 1
    return mpjpe/n


def find_best_mpjpe(pred, gt_skeletons):
    best_mpjpe = 1e9
    for gt in gt_skeletons:
        mpjpe = calculate_mpjpe(pred, gt)
        best_mpjpe = min(best_mpjpe, mpjpe)
    return best_mpjpe


def main():
    track_jsons = os.listdir(track_dir)

    mpjpe_sum = 0

    for track_json in track_jsons:
        track_path = os.path.join(track_dir, track_json)
        track_mpjpe = 0
        with open(track_path, "r") as file:
            # Load the JSON data from the file
            track_data = json.load(file)
            frames = track_data["frames"]
            poses = track_data["poses"]

        frames_used = 0
        for i, frame in enumerate(frames):
            gt_poses = get_frame_poses(os.path.join(gt_dir, "body3DScene_"+str(frame+frame_offset).zfill(8)+".json"))
            if gt_poses is None:
                continue
            pred_pose = poses[i]
            track_mpjpe += find_best_mpjpe(pred_pose, gt_poses)
            frames_used += 1
        if track_mpjpe/frames_used > 50:
            continue
        mpjpe_sum += track_mpjpe/frames_used
        print(track_mpjpe/frames_used)

    print(mpjpe_sum/len(track_jsons))


if __name__=="__main__":
    main()