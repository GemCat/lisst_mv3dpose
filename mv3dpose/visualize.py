import sys
sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from mv3dpose.tracking import Track
from mpl_toolkits.mplot3d import axes3d, Axes3D
from os.path import join, isfile
from mv3dpose.config import Config
from mv3dpose.load_functions import load_cameras
from mv3dpose.plot_functions import colors, LIMBS, isclose
from os import listdir
from tqdm import tqdm
import math

if __name__ == '__main__':

    # ~~~~~ LOAD SETTINGS ~~~~~
    # the first argument is the path to dataset dir  
    dataset_dir = sys.argv[1]
    conf = Config(dataset_dir, vis = True)

    calib = load_cameras(conf)
    tracks = [json.load(open(join(conf.trk_dir, f))) for f in sorted(listdir(conf.trk_dir))]
    n_tracks = len(tracks)
    valid_frames = list(range(conf.vis_frames))
    cameras = range(conf.n_cameras)
    print("#tracks", n_tracks)

    # -- create lookups --
    tracks_by_frame = {}
    pose_by_track_and_frame = {}
    for frame in valid_frames:
        assert frame not in tracks_by_frame
        tracks_by_frame[frame] = []
        for tid, track in enumerate(tracks):
            frames = track['frames']
            poses = track['poses']
            for i, t in enumerate(frames):
                if t > frame:
                    break
                elif t == frame:
                    tracks_by_frame[frame].append(tid)
                    pose_by_track_and_frame[tid, frame] = poses[i]

    # ====================================
    # ~~~~ PLOT FRAMES ~~~~
    # ====================================
    colors = colors(n_tracks)

    for i, frame in tqdm(enumerate(valid_frames)):

        fig = plt.figure(figsize=(16, 12))
        H = 2 if conf.n_cameras < 8 else 3
        W = int(math.ceil(conf.n_cameras / H))
        fname = join(conf.vis_dir, 'frame%09d.png' % i)
        tracks_on_frame = tracks_by_frame[frame]

        for camnbr, cid in enumerate(cameras):

            camera_img_dir = join(conf.img_dir, 'camera%02d' % cid)
            fr = "frame%02d_" % cid
            img_file = join(camera_img_dir, (f'{fr}%09d.' % frame) + conf.img_file_type)
            assert isfile(img_file), img_file
            im = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape

            cam = calib[cid]
            ax = fig.add_subplot(H, W, camnbr+1)
            ax.axis('off')
            ax.set_xlim([0, w])
            ax.set_ylim([h, 0])
            ax.imshow(im)

            for tid in tracks_on_frame:
                color = colors[tid%len(colors)]
                pose3d = pose_by_track_and_frame[tid, frame]

                # we need to mask over None
                assert len(pose3d) == 24
                mask = [True] * 24
                for jid in range(24):
                    if pose3d[jid] is None:
                        pose3d[jid] = [0, 0, 0]
                        mask[jid] = False
                    else:
                        mm = np.mean(pose3d[jid])
                        if isclose(0., mm):
                            mask[jid] = False

                pose3d = np.array(pose3d, dtype=np.float32)

                pose2d = cam.projectPoints(pose3d)
                for jid in range(24):
                    if mask[jid]:
                        x, y = pose2d[jid]
                        ax.scatter(x, y, color=color)

                for a, b in LIMBS:
                    if mask[a] and mask[b]:
                        x1, y1 = pose2d[a]
                        x2, y2 = pose2d[b]
                        ax.plot([x1, x2], [y1, y2], color=color)

        # 3D plot ================
        if conf.plot3d:
            ax = fig.add_subplot(H, W, conf.n_cameras + 1, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            min_x, min_y, min_z = 3*[0]
            max_x, max_y, max_z = 3*[0]

            for tid in tracks_on_frame:
                
                color = colors[tid%len(colors)]
                pose3d = pose_by_track_and_frame[tid, frame]

                mask = [True] * 24
                for jid in range(24):
                    if pose3d[jid] is None:
                        mask[jid] = False
                    else:
                        x, y, z = pose3d[jid]
                        if np.isclose(x, .0) and np.isclose(y, .0) and np.isclose(z, .0):
                            mask[jid] = False
                        else:
                            ax.scatter(x, y, z, color=color)

                for a, b in LIMBS:
                    if mask[a] and mask[b]:
                        x1, y1, z1 = pose3d[a]
                        x2, y2, z2 = pose3d[b]
                        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=4)
                        max_x = max(max_x, x1, x2)
                        min_x = min(min_x, x1, x2)
                        max_y = max(max_y, y1, y2)
                        min_y = min(min_y, y1, y2)
                        max_z = max(max_z, z1, z2)
                        min_z = min(min_z, z1, z2)
                
                ax.set_xlim([min_x-50, max_x+50])
                ax.set_ylim([min_y-50, max_y+50])
                ax.set_zlim([min_z-50, max_z+50])

        # ============= (end) 3D plot

        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
