import logging
import logging.handlers
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from from_super_glue_pretrained.superglue import SuperGlue
from from_super_glue_pretrained.superpoint import SuperPoint
from homography_superpoint_dataset import ParisHomographySuperPointDataset
from utils import create_log_folder, timestamp, calculate_prec_recall


def point_in_image(img_shape, point):
    if point[0] < 0 or point[1] < 0:
        return False

    if point[0] > img_shape[1] or point[1] > img_shape[0]:
        return False

    return True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def train(debug_mode):
    device = torch.device("cuda:0")

    superpoint_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        # 'keypoint_threshold': 0.005,
        'keypoint_threshold': 0.00,
        'max_keypoints': 512,
        'remove_borders': 4,
    }

    superpoint = SuperPoint(superpoint_config).eval().to(device)

    superglue_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    superglue = SuperGlue(superglue_config).to(device)
    max_dist = 1

    homography_train_dataset = ParisHomographySuperPointDataset(max_dist, superpoint, device)
    dataloader_train = DataLoader(homography_train_dataset,
                                  shuffle=True, num_workers=0, batch_size=10,
                                  drop_last=True,
                                  collate_fn=homography_train_dataset.collater)

    validation_folder_names = ["pantheon"]

    homography_validation_dataset = ParisHomographySuperPointDataset(max_dist, superpoint, device, validation_folder_names)
    dataloader_validation = DataLoader(homography_validation_dataset, num_workers=0, batch_size=1,
                                  collate_fn=homography_validation_dataset.collater)

    optimizer = optim.Adam(superglue.parameters(), lr=1e-4)

    log_folder_name = "debug_training_submission"
    experiment_folder_path = create_log_folder(Path(__file__).parent / "logs", log_folder_name, str(timestamp()))
    number_of_epochs = 1000

    experiment_folder_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=experiment_folder_path)

    # Set up a specific logger with our desired output level
    logger = logging.getLogger('MyLogger')
    logger.setLevel(logging.INFO)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler("{0}/{1}.txt".format(experiment_folder_path, log_folder_name))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.info("start")

    logger.info("superglue config:")
    logger.info(superglue_config)
    logger.info("superpoint config:")
    logger.info(superpoint_config)
    logger.info("max_dist {}".format(max_dist))

    for epoch_index in range(number_of_epochs):

        precision_list = []
        recall_list = []
        loss_list = []
        superglue.train()

        for index, (data, all_matches, gt_matches) in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            scores = superglue.forward_supervised(data)

            # calculate loss
            loss_list_for_calculate_batch_loss = []
            for batch_idx in range(len(all_matches)):
                item_loss = -1.0 * scores[batch_idx, all_matches[batch_idx][0], all_matches[batch_idx][1]]
                loss_list.append(item_loss.mean().item())
                loss_list_for_calculate_batch_loss.append(item_loss)

            loss = torch.cat(loss_list_for_calculate_batch_loss).mean()
            loss.backward()
            gr_clip = 0.1
            torch.nn.utils.clip_grad_norm_(superglue.parameters(), gr_clip)
            optimizer.step()

            # calculate accuracy
            out = superglue.evaluate_matching(scores, superglue.config['match_threshold'])

            batch_prec_list, batch_recall_list, last_superglue_match = calculate_prec_recall(gt_matches, out)

            precision_list.extend(batch_prec_list)
            recall_list.extend(batch_recall_list)

            if index % 5 == 0:
                logger.info('prec {} recall {} loss {}'.format(precision_list[-1], recall_list[-1], loss_list[-1]))

                if debug_mode:
                    scores_0 = out["matching_scores0"].detach().cpu().numpy().squeeze()
                    draw_image_projection_match(homography_train_dataset.image_kp_projected_to_warped,
                                                homography_train_dataset.image,
                                                data['keypoints0'][-1],
                                                homography_train_dataset.warped, False)

                    draw_superpoint_and_projected_match(homography_train_dataset.image_kp_projected_to_warped,
                                                        homography_train_dataset.image, data['keypoints0'][-1], data['keypoints1'][-1],
                                                        gt_matches[-1], homography_train_dataset.warped,
                                                        homography_train_dataset.projected_to_warped_and_superpoints_dist, False)

                    draw_superglue_match(homography_train_dataset.image,
                                         data['keypoints0'][-1],
                                         last_superglue_match,
                                         data['keypoints1'][-1], homography_train_dataset.warped,
                                         homography_train_dataset.projected_to_warped_and_superpoints_dist, scores_0, False)

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_loss = np.mean(loss_list)

        writer.add_scalar('epoch/train/mean_loss', mean_loss, epoch_index)
        writer.add_scalar('epoch/train/mean_recall', mean_recall, epoch_index)
        writer.add_scalar('epoch/train/mean_precision', mean_precision, epoch_index)

        logger.info('training epoch {} mean prec {} mean recall {} mean loss {}'.format(epoch_index, mean_precision, mean_recall,
                                                                               mean_loss))

        # calculate validation performance
        superglue.eval()
        mean_loss, mean_precision, mean_recall = calculate_validation_results(dataloader_validation, superglue)

        torch.save(superglue.state_dict(), experiment_folder_path / "model_{}.pth".format(epoch_index))
        writer.add_scalar('epoch/validation/mean_loss', mean_loss, epoch_index)
        writer.add_scalar('epoch/validation/mean_recall', mean_recall, epoch_index)
        writer.add_scalar('epoch/validation/mean_precision', mean_precision, epoch_index)
        logger.info('validation epoch {} mean prec {} mean recall {} mean loss {}'.format(epoch_index, mean_precision, mean_recall, mean_loss))


def calculate_validation_results(dataloader_validation, superglue):
    precision_list = []
    recall_list = []
    loss_list = []
    for index, (data, all_matches, gt_matches) in enumerate(tqdm(dataloader_validation)):
        with torch.no_grad():
            scores = superglue.forward_supervised(data)

        for batch_idx in range(len(all_matches)):
            item_loss = -1.0 * scores[batch_idx, all_matches[batch_idx][0], all_matches[batch_idx][1]]
            loss_list.append(item_loss.mean().item())

        out = superglue.evaluate_matching(scores, superglue.config['match_threshold'])

        batch_prec_list, batch_recall_list, last_superglue_match = calculate_prec_recall(gt_matches, out)

        precision_list.extend(batch_prec_list)
        recall_list.extend(batch_recall_list)

    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_loss = np.mean(loss_list)
    return mean_loss, mean_precision, mean_recall


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
    for match in match_arg.transpose():
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


if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    train(debug_mode=False)
