import torch
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import datetime

class HomographyDataLoader(Dataset):

    def __init__(self):
        self.nfeatures = 50
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

        self.image = cv2.imread("image_goes_here")

        self.height, self.width = self.image.shape[:2]
        self.max_size = max(self.height, self.width)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        print(f"seed = {seed}")
        np.random.seed(seed)
    def __getitem__(self, item):

        image = self.image
        sift = self.sift
        max_size = self.max_size
        width = self.width
        height = self.height

        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        M = cv2.getPerspectiveTransform(corners, corners + warp)

        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))
        #print(warp)
        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        kp1 = kp1[:self.nfeatures]
        kp2 = kp2[:self.nfeatures]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        kp1_np = kp1_np[:self.nfeatures, :]
        kp2_np = kp2_np[:self.nfeatures, :]
        descs1 = descs1[:self.nfeatures, :]
        descs2 = descs2[:self.nfeatures, :]

        matched = self.matcher.match(descs1, descs2)

        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]

        dists = cdist(kp1_projected, kp2_np)

        for mm in matched:
            dd = dists[mm.queryIdx, mm.trainIdx]
            print(dd)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        visualize = False
        if visualize:
            matches_dmatch = []
            for idx in range(matches.shape[0]):
                dmatch = cv2.DMatch(matches[idx], min2[matches[idx]], 0.0)
                print(f"Match {matches[idx]} {min2[matches[idx]]} dist={dists[matches[idx], min2[matches[idx]]]}")
                matches_dmatch.append(dmatch)
            out = cv2.drawMatches(image, kp1, warped, kp2, matches_dmatch, None)
            cv2.imshow('a', out)
            cv2.waitKey(0)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)
        '''
        for idx in range(all_matches.shape[1]):
            pt1 = all_matches[0, idx]
            pt2 = all_matches[1, idx]
            if pt1 != self.nfeatures and pt2 != self.nfeatures:
                print(f"match: {dists[pt1, pt2]} | {pt2} {np.argmin(dists[pt1, :])} | {pt1} {np.argmin(dists[:, pt2])}")
            else:
                print(f"no match {pt1} {pt2}")
        '''
        return {'kp1': kp1_np / max_size, 'kp2': kp2_np / max_size, 'descs1': descs1 / 256., 'descs2': descs2 / 256., 'matches': all_matches}

    def __len__(self):
        return 320


def collater(data):
    kp1 = np.concatenate([np.expand_dims(s['kp1'], axis=0) for s in data], axis=0)
    kp2 = np.concatenate([np.expand_dims(s['kp2'], axis=0) for s in data], axis=0)

    descs1 = np.concatenate([np.expand_dims(s['descs1'], axis=0) for s in data], axis=0)
    descs2 = np.concatenate([np.expand_dims(s['descs2'], axis=0) for s in data], axis=0)

    matches = [s['matches'] for s in data]

    return (torch.from_numpy(kp1).float(),
            torch.from_numpy(kp2).float(),
            torch.from_numpy(descs1).float(),
            torch.from_numpy(descs2).float(),
            matches)
