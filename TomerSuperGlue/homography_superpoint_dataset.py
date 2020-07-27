from os import listdir
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

from utils import get_matching_points_by_distance


class HomographySuperPointDataset(Dataset):
    def __init__(self, max_distance, superpoint, device):
        self.viz = False
        self.superpoint = superpoint
        self.max_distance = max_distance
        self.device = device

        self.image_result = None
        self.warped_result = None
        self.image_kp_projected_to_warped = None
        self.projected_to_warped_and_superpoints_dist = None

    def get_item_path(self, idx):
        raise NotImplemented

    def __getitem__(self, idx):
        image_rgb = cv2.imread(self.get_item_path(idx))
        self.image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        with torch.no_grad():
            M, self.warped = self.get_warped_image(self.image)
            image = torch.unsqueeze(torch.from_numpy(self.image.astype("float32") / 255.), 0)
            warped = torch.unsqueeze(torch.from_numpy(self.warped.astype("float32") / 255.), 0)

            self.image_result = self.superpoint({'image': image.unsqueeze(0).to(self.device)})
            self.warped_result = self.superpoint({'image': warped.unsqueeze(0).to(self.device)})

            all_matches, gt_match, self.image_kp_projected_to_warped, self.projected_to_warped_and_superpoints_dist = \
                self.get_gt_matching(M, self.image_result, self.warped_result)

        if self.viz:
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.imshow(warped, cmap='gray')
            plt.show()

        return image, self.image_result, warped, self.warped_result, all_matches, gt_match


    @staticmethod
    def collater(input_data):
        images = [item[0] for item in input_data]
        img_results = [item[1] for item in input_data]
        warpeds = [item[2] for item in input_data]
        warped_results = [item[3] for item in input_data]

        des0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['descriptors']])
        sco0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['scores']])
        key0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['keypoints']])
        des1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['descriptors']])
        sco1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['scores']])
        key1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['keypoints']])

        images_shapes = [(image.shape[-2:], warped.shape[-2:]) for image, warped in zip(images, warpeds)]

        data = {'descriptors0': des0,
                'scores0': sco0,
                'keypoints0': key0,
                'descriptors1': des1,
                'scores1': sco1,
                'keypoints1': key1,
                'images_shapes': images_shapes}

        all_matches = [item[4] for item in input_data]
        gt_matches = [item[5] for item in input_data]
        return data, all_matches, gt_matches


    @staticmethod
    def get_warped_image(image):
        height, width = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))
        return M, warped

    @staticmethod
    def get_gt_matching(M, img_result, warped_result):
        img_kp = img_result['keypoints'][0].detach().cpu().numpy()
        warped_kp = warped_result['keypoints'][0].detach().cpu().numpy()
        image_kp_projected_to_warped = cv2.perspectiveTransform(img_kp.reshape((1, -1, 2)),
                                                                M)[0, :, :]
        projected_to_warped_and_superpoints_dist = cdist(image_kp_projected_to_warped, warped_kp)
        # match by applying super point to the second image and calculate projection and distance
        gt_match = get_matching_points_by_distance(image_kp_projected_to_warped,
                                                   projected_to_warped_and_superpoints_dist,
                                                   max_dist=1)
        missing1 = np.setdiff1d(np.arange(image_kp_projected_to_warped.shape[0]), gt_match[0])
        missing2 = np.setdiff1d(np.arange(warped_kp.shape[0]), gt_match[1])
        MN2 = np.concatenate(
            [missing1[np.newaxis, :], (len(warped_kp)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate(
            [(len(img_kp)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([gt_match, MN2, MN3], axis=1)

        return all_matches, gt_match, image_kp_projected_to_warped, projected_to_warped_and_superpoints_dist

    def __len__(self):
        raise NotImplemented


class ParisHomographySuperPointDataset(HomographySuperPointDataset):
    def __init__(self, max_distance, superpoint, device, folder_names=None):
        super().__init__(max_distance, superpoint, device)
        if folder_names is None:
            folder_names = ["eiffel", "moulinrouge", "defense", "general", "louvre", "invalides", "triomphe", "sacrecoeur",
             "pompidou"]
        paris_dataset_path = Path(__file__).parent.parent / "data" / "paris"
        self.base_path = [str(paris_dataset_path / name) for name in folder_names]
        print(self.base_path)

        self.ex_names = []
        for p in self.base_path:
            for j in listdir(p):
                self.ex_names.append(str(Path(p) / j))

        # delete bad images
        # for path in tqdm(self.ex_names):
        #     img = cv2.imread(path)
        #     if cv2.imread(path) is None:
        #         print("removing {}".format(path))
        #         os.remove(path)
        #
        #     else:
        #         if img.shape[0] < 200 or img.shape[1] < 200:
        #             print("removing {} because of shape".format(path))
        #             os.remove(path)

    def get_item_path(self, idx):
        return self.ex_names[idx]

    def __len__(self):
        return len(self.ex_names)


class OxfordHomographySuperPointDataset(HomographySuperPointDataset):
    def __init__(self, max_distance, superpoint, device):
        super().__init__(max_distance, superpoint, device)
        oxford_dataset_path = Path(__file__).parent.parent / "data" / "oxford_dataset"

        self.ex_names = []
        for path in oxford_dataset_path.iterdir():
            self.ex_names.append(str(path))

        # delete bad images
        # for path in tqdm(self.ex_names):
        #     img = cv2.imread(path)
        #     if cv2.imread(path) is None:
        #         print("removing {}".format(path))
        #         os.remove(path)
        #
        #     if img.shape[0] < 200 or img.shape[1] < 200:
        #         print("removing {} because of shape".format(path))
        #         os.remove(path)

    def get_item_path(self, idx):
        return self.ex_names[idx]

    def __len__(self):
        return len(self.ex_names)


class HomographySuperPointDatasetTriplets(HomographySuperPointDataset):
    def __init__(self, max_distance, superpoint, device):
        super().__init__(max_distance, superpoint, device)

    def __getitem__(self, idx):
        image_rgb = cv2.imread(self.get_item_path(idx))
        image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        with torch.no_grad():

            M, warped = self.get_warped_image(image)
            M2, warped2 = self.get_warped_image(image)

            image = torch.unsqueeze(torch.from_numpy(image.astype("float32") / 255.), 0)
            warped = torch.unsqueeze(torch.from_numpy(warped.astype("float32") / 255.), 0)
            warped2 = torch.unsqueeze(torch.from_numpy(warped2.astype("float32") / 255.), 0)

            img_result = self.superpoint({'image': image.unsqueeze(0).to(self.device)})
            warped_result = self.superpoint({'image': warped.unsqueeze(0).to(self.device)})
            warped2_result = self.superpoint({'image': warped2.unsqueeze(0).to(self.device)})

            all_matches, gt_matches, _, _ = self.get_gt_matching(M, img_result, warped_result)
            all_matches2, gt_matches2, _, _ = self.get_gt_matching(M2, img_result, warped2_result)

        if self.viz:
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.imshow(warped, cmap='gray')
            plt.show()

        images_shapes = (image.shape[-2:], warped.shape[-2:], warped2.shape[-2:])
        return img_result, warped_result, warped2_result, images_shapes, all_matches, gt_matches, all_matches2, gt_matches2

    @staticmethod
    def collater(input_data):
        img_results = [item[0] for item in input_data]
        warped_results = [item[1] for item in input_data]
        warped2_results = [item[2] for item in input_data]
        images_shapes = [item[3] for item in input_data]

        des0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['descriptors']])
        sco0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['scores']])
        key0 = torch.cat([torch.unsqueeze(a, 0) for result in img_results for a in result['keypoints']])
        des1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['descriptors']])
        sco1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['scores']])
        key1 = torch.cat([torch.unsqueeze(a, 0) for result in warped_results for a in result['keypoints']])
        des2 = torch.cat([torch.unsqueeze(a, 0) for result in warped2_results for a in result['descriptors']])
        sco2 = torch.cat([torch.unsqueeze(a, 0) for result in warped2_results for a in result['scores']])
        key2 = torch.cat([torch.unsqueeze(a, 0) for result in warped2_results for a in result['keypoints']])

        data = {'descriptors0': des0, 'scores0': sco0, 'keypoints0': key0,
                'descriptors1': des1, 'scores1': sco1, 'keypoints1': key1,
                'descriptors2': des2, 'scores2': sco2, 'keypoints2': key2,
                'images_shapes': images_shapes}

        all_matches = [item[4] for item in input_data]
        gt_matches = [item[5] for item in input_data]
        all_matches2 = [item[6] for item in input_data]
        gt_matches2 = [item[7] for item in input_data]
        return data, all_matches, gt_matches, all_matches2, gt_matches2


class ParisHomographySuperPointDatasetTriplets(HomographySuperPointDatasetTriplets):
    def __init__(self, folder_names, max_distance, superpoint, device):
        super().__init__(max_distance, superpoint, device)
        paris_dataset_path = Path(__file__).parent.parent / "data" / "paris"
        self.base_path = [str(paris_dataset_path / name) for name in folder_names]
        print(self.base_path)

        self.ex_names = []
        for p in self.base_path:
            for j in listdir(p):
                self.ex_names.append(str(Path(p) / j))

        # delete bad images
        # for path in tqdm(self.ex_names):
        #     img = cv2.imread(path)
        #     if cv2.imread(path) is None:
        #         print("removing {}".format(path))
        #         os.remove(path)
        #
        #     else:
        #         if img.shape[0] < 200 or img.shape[1] < 200:
        #             print("removing {} because of shape".format(path))
        #             os.remove(path)

    def get_item_path(self, idx):
        return self.ex_names[idx]

    def __len__(self):
        return len(self.ex_names)


class OxfordHomographySuperPointDatasetTriplets(HomographySuperPointDatasetTriplets):
    def __init__(self, max_distance, superpoint, device):
        super().__init__(max_distance, superpoint, device)
        oxford_dataset_path = Path(__file__).parent.parent / "data" / "oxford_dataset"

        self.ex_names = []
        for path in oxford_dataset_path.iterdir():
            self.ex_names.append(str(path))

        # delete bad images
        # for path in tqdm(self.ex_names):
        #     img = cv2.imread(path)
        #     if cv2.imread(path) is None:
        #         print("removing {}".format(path))
        #         os.remove(path)
        #
        #     if img.shape[0] < 200 or img.shape[1] < 200:
        #         print("removing {} because of shape".format(path))
        #         os.remove(path)

    def get_item_path(self, idx):
        return self.ex_names[idx]

    def __len__(self):
        return len(self.ex_names)
