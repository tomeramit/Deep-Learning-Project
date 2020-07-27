import logging
import logging.handlers
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from from_super_glue_pretrained.superglue import SuperGlue
from from_super_glue_pretrained.superpoint import SuperPoint
from homography_superpoint_dataset import OxfordHomographySuperPointDataset, ParisHomographySuperPointDataset
from utils import create_log_folder, timestamp, calculate_prec_recall


def point_in_image(img_shape, point):
    if point[0] < 0 or point[1] < 0:
        return False

    if point[0] > img_shape[1] or point[1] > img_shape[0]:
        return False

    return True


def eval(dataset, superpoint_config, superglue_config, logs_folder_name, logger):
    device = torch.device("cuda:0")

    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    superpoint = SuperPoint(superpoint_config).eval().to(device)

    superglue = SuperGlue(superglue_config).eval().to(device)
    superglue.load_weights(superglue_config['weights'])

    homography_dataset = dataset(1, superpoint, device)
    dataloader_train = DataLoader(homography_dataset,
                                  shuffle=False, num_workers=0, batch_size=1,
                                  collate_fn=dataset.collater)

    if len(Path(superglue_config["weights"]).parts) > 1:
        folder_name = "{}_{}".format(Path(superglue_config["weights"]).parts[-2], Path(superglue_config["weights"]).parts[-1])
    else:
        folder_name = superglue_config["weights"]

    experiment_folder_path = create_log_folder(Path(__file__).parent / logs_folder_name,
                                               "eval_{}".format(folder_name),
                                               str(timestamp()))

    experiment_folder_path.mkdir(parents=True, exist_ok=True)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler("{0}/{1}.txt".format(experiment_folder_path, folder_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    logger.info("superglue config:")
    logger.info(superglue_config)
    logger.info("superpoint config:")
    logger.info(superpoint_config)
    logger.info("dataset type: {}".format(type(homography_dataset)))

    precision_list = []
    recall_list = []
    loss_list = []

    for index, (data, all_matches, gt_matches) in enumerate(tqdm(dataloader_train)):

        with torch.no_grad():
            scores = superglue.forward_supervised(data)

        # calculate loss
        loss_list_for_calculate_batch_loss = []
        for batch_idx in range(len(all_matches)):
            item_loss = -1.0 * scores[batch_idx, all_matches[batch_idx][0], all_matches[batch_idx][1]]
            loss_list_for_calculate_batch_loss.append(item_loss)
            loss_list.append(item_loss.mean().item())

        # calculate accuracy
        out = superglue.evaluate_matching(scores, superglue.config['match_threshold'])

        batch_prec_list, batch_recall_list, last_superglue_match = calculate_prec_recall(gt_matches, out)

        precision_list.extend(batch_prec_list)
        recall_list.extend(batch_recall_list)

            # if debug_mode:

            #     scores_0 = out["matching_scores0"].detach().cpu().numpy().squeeze()
            #     draw_image_projection_match(image_kp_projected_to_warped, img, img_kp, wrp, verbose_mode)
            #     draw_superpoint_and_projected_match(image_kp_projected_to_warped, img, img_kp, warped_kp, gt_match, wrp,
            #                                         projected_to_warped_and_superpoints_dist, verbose_mode)
            #     draw_superglue_match(img, img_kp, superglue_match, warped_kp, wrp,
            #                          projected_to_warped_and_superpoints_dist, scores_0, verbose_mode)

    mean_precision = np.array(precision_list).mean()
    mean_recall = np.array(recall_list).mean()
    mean_loss = np.array(loss_list).mean()

    logger.info('eval: mean prec {} mean recall {} mean loss {}'.format(mean_precision, mean_recall, mean_loss))
    logger.removeHandler(fileHandler)


def get_superglue_matching(matches_0, matches_1):
    superglue_match = []
    for index, match_index_0 in enumerate(matches_0):
        if match_index_0 == -1:
            continue
        match_index_1 = matches_1[match_index_0]

        if matches_0[match_index_1] != match_index_0:
            raise Exception()
        superglue_match.append([index, match_index_0])
    return superglue_match


def draw_superglue_match(img, img_kp, superglue_match, warped_kp, wrp, dist, scores_0, verbose):
    img_kp_objects = [cv2.KeyPoint(i[0], i[1], 20) for i in img_kp]
    warped_kp_objects = [cv2.KeyPoint(i[0], i[1], 20) for i in warped_kp]

    superglue_match_objects = []
    for index_0, index_1 in superglue_match:
        if verbose:
            print(f"Match {index_0} {index_1} dist={dist[index_0, index_1]} score={scores_0[index_0]}")
        superglue_match_objects.append(cv2.DMatch(index_0, index_1, 0.0))

    draw_img = cv2.drawMatches(img, img_kp_objects, wrp, warped_kp_objects, superglue_match_objects, None,
                               flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("image kp and warped kp superglue matching")
    plt.show()


def draw_superpoint_and_projected_match(image_kp_projected_to_warped, img, img_kp, warped_kp, match_arg, wrp, d, verbose):
    img_kp_objects = [cv2.KeyPoint(i[0], i[1], 20) for i in img_kp]
    image_kp_objects_projected_to_warped = [cv2.KeyPoint(i[0], i[1], 20) for i in
                                            image_kp_projected_to_warped]
    warped_kp_objects = [cv2.KeyPoint(i[0], i[1], 20) for i in warped_kp]

    draw_img = cv2.drawKeypoints(wrp, image_kp_objects_projected_to_warped, None,
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("warped image with image kp projected to warped")
    plt.show()

    draw_img = cv2.drawKeypoints(wrp, warped_kp_objects, None,
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("warped image with warped kp")
    plt.show()

    matches_dmatch = []
    if verbose:
        print("Match of image and projected image by distance")
    for match in match_arg:
        dmatch = cv2.DMatch(match[0], match[1], 0.0)
        if verbose:
            print(f"Match {match[0]} {match[1]} dist={d[match[0], match[1]]}")
        matches_dmatch.append(dmatch)

    draw_img = cv2.drawMatches(img, img_kp_objects, wrp, warped_kp_objects, matches_dmatch, None,
                               flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("image kp and warped kp gt matching")
    plt.show()


def draw_image_projection_match(image_kp_projected_to_warped, img, img_kp, wrp, verbose):
    img_kp_objects = [cv2.KeyPoint(i[0], i[1], 20) for i in img_kp]
    image_kp_objects_projected_to_warped = [cv2.KeyPoint(i[0], i[1], 20) for i in
                                            image_kp_projected_to_warped]

    draw_img = cv2.drawKeypoints(img, img_kp_objects, None,
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("image with image kp")
    plt.show()
    plt.close()

    draw_img = cv2.drawKeypoints(wrp, image_kp_objects_projected_to_warped, None,
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.imshow(draw_img)
    plt.title("warped image with image kp projected to warped")
    plt.show()
    relevant_point_indexes = []
    for index, point in enumerate(image_kp_projected_to_warped):
        if point_in_image(img.shape, point):
            relevant_point_indexes.append(index)

    relevant_point_indexes = np.array(relevant_point_indexes)
    matches_dmatch = []
    if verbose:
        print("Match of image and projected image")
    for point_index in relevant_point_indexes:
        dmatch = cv2.DMatch(point_index, point_index, 0.0)
        if verbose:
            print(f"Match {point_index} {point_index} dist={0}")
        matches_dmatch.append(dmatch)
    draw_img = cv2.drawMatches(img, img_kp_objects, wrp, image_kp_objects_projected_to_warped, matches_dmatch, None,
                               flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    plt.imshow(draw_img)
    plt.title("image kp and image kp projected to warped gt matching")
    plt.show()

    return image_kp_objects_projected_to_warped


def main():
    superpoint_config_default = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 512,
        'remove_borders': 4,
    }

    superglue_config_default = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20
    }
    #
    # list_of_weights = [
    #     'outdoor'
    #     'logs/20200723_163657 - train_long_max_dist_1_max_keypoints_1024/model_50.pth',
    #     'logs/trained_supervised/gnn_hyper_superglue_16_64_no_bias_idxs_c_01_lr_1-4_0_epoch_186.pt',
    #     'logs/trained_supervised/gnn_hyper_superglue_16_64_no_bias_idxs_c_01_lr_1-4_0_epoch_400.pt',
    #     'logs/trained_supervised_with_my_fix/gnn_hyper_sagie_model_with_my_gt_match_function0_epoch_146.pt'
    # ]

    # compare long trainings
    # list_of_weights = [
    #     'logs/20200722_084338 - train_long/model_100.pth',
    #     'logs/20200722_084338 - train_long/model_150.pth',
    #     'logs/20200722_084338 - train_long/model_190.pth'
    # ]

    # compare more keypoints
    list_of_weights = [
        # 'from_super_glue_pretrained/weights/superglue_outdoor.pth',
        'logs/20200726_142634 - training_submission/model_62.pth'
    ]

    # Set up a specific logger with our desired output level
    logger = logging.getLogger('MyLogger')
    logger.setLevel(logging.INFO)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    for weight in list_of_weights:
        superglue_config = superglue_config_default.copy()
        superglue_config['weights'] = weight
        eval(OxfordHomographySuperPointDataset, superpoint_config_default, superglue_config, "logs_supervised_homography_tomer", logger)
        eval(ParisHomographySuperPointDataset, superpoint_config_default, superglue_config, "logs_supervised_homography_tomer",
             logger)


if __name__ == "__main__":
    main()
