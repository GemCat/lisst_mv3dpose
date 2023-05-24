import numpy as np
import json
from os.path import isdir, isfile, join


class MultiOpenPoseKeypoints:

    def __init__(self, pe_kpts):
        """
        :param pe_kpts: {list}
        """
        self.pe_kpts = pe_kpts

    def predict(self, frame):
        """
        :param frame:
        :return:
        """
        result = []
        for pe in self.pe_kpts:
            pred = pe.predict(frame)
            result.append(pred)
        return result


class OpenPoseKeypoints:

    def __init__(self, naming, loc):
        """
        :param naming:
        :param loc:
        """
        assert isdir(loc), loc
        self.loc = loc
        self.naming = naming

    def predict(self, frame):
        """
        :param frame:
        :return:
        """
        if "dance" in self.loc:
            fr = frame + 100
        else:
            fr = frame
        fname = join(self.loc, self.naming + '_keypoints.json') % (fr, )
        assert isfile(fname), fname
        with open(fname, 'r') as f:
            kp = json.load(f)

        ours_vs_openpose = np.array([
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
            (5, 5), (6, 6), (7, 7),
            (8, 9), (9, 10), (10, 11),
            (11, 12), (12, 13), (13, 14),
            (14, 15), (15, 16), (16, 17), (17, 18),
            (18, 19), (19, 20), (20, 21),
            (21, 22), (22, 23), (23, 24),
            (24, 34), (25, 29), (26, 37),  # left hand, thumb, middle
            (27, 55), (28, 50), (29, 58)  # right hand, thumb, middle
        ])

        # J = 18
        J = 30

        results = []
        for person in kp['people']:
            OUR = ours_vs_openpose[:, 0]
            OP = ours_vs_openpose[:, 1]
            our_person = np.empty((J, 3), np.float32)
            body = np.reshape(person['pose_keypoints_2d'], (-1, 3))
            lhand = np.reshape(person['hand_left_keypoints_2d'], (-1, 3))
            rhand = np.reshape(person['hand_right_keypoints_2d'], (-1, 3))
            kps = np.concatenate((body,lhand,rhand), axis=0)
            our_person[OUR] = kps[OP]

            for i, (x, y, v) in enumerate(our_person):
                if v < 0.001:
                    our_person[i, 0] = -1
                    our_person[i, 1] = -1
                    our_person[i, 2] = -1

            results.append(our_person)

        return results
