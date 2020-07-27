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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import create_log_folder, timestamp, calculate_matches

from from_super_glue_pretrained.superglue import SuperGlue
from from_super_glue_pretrained.superpoint import SuperPoint
from homography_superpoint_dataset import ParisHomographySuperPointDatasetTriplets, ParisHomographySuperPointDataset


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


def train(train, debug_mode, verbose_mode):
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
    # superglue.apply(weights_init)
    max_dist = 1

    train_folder_names = ["eiffel", "moulinrouge", "defense", "general", "louvre", "invalides", "triomphe", "sacrecoeur",
                    "pompidou"]

    homography_train_dataset = ParisHomographySuperPointDatasetTriplets(train_folder_names, max_dist, superpoint, device)
    dataloader_train = DataLoader(homography_train_dataset,
                                  shuffle=True, num_workers=0, batch_size=2,
                                  drop_last=True,
                                  collate_fn=homography_train_dataset.collater)

    validation_folder_names = ["pantheon"]

    homography_validation_dataset = ParisHomographySuperPointDataset(max_dist, superpoint, device, validation_folder_names)
    dataloader_validation = DataLoader(homography_validation_dataset, num_workers=0, batch_size=1,
                                  collate_fn=homography_validation_dataset.collater)

    optimizer = optim.Adam(superglue.parameters(), lr=1e-4)

    log_folder_name = "debug_train_unsupervised"
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

    mse = torch.nn.MSELoss()

    for epoch_index in range(number_of_epochs):

        precision_list = []
        recall_list = []
        validation_loss_list = []
        triplet_loss_list = []
        cycle_loss_list = []
        id_loss_list = []
        batch_total_loss_list = []
        superglue.train()

        for index, (data, all_matches, gt_matches, all_matches2, gt_matches2) in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()

            item_loss_list, cycle_loss = calculate_cycle_loss(all_matches, data, gt_matches, superglue,
                                              superpoint_config["max_keypoints"])
            cycle_loss_list.extend(item_loss_list)


            item_loss_list, id_loss = calc_id_loss(all_matches, data, superglue)
            id_loss_list.extend(item_loss_list)


            abc_scores, ac_scores = superglue.forward_unsupervised(data)
            triplet_loss = mse(abc_scores, ac_scores)
            triplet_loss_list.extend([mse(abc_score, ac_score).item() for abc_score, ac_score in zip(abc_scores, ac_scores)])

            loss = id_loss + cycle_loss + triplet_loss
            batch_total_loss_list.append(loss.item())
            loss.backward()
            # gr_clip = 0.1
            # torch.nn.utils.clip_grad_norm_(superglue.parameters(), gr_clip)
            optimizer.step()

            # calculate validation loss
            for batch_idx in range(len(all_matches2)):
                item_loss = -1.0 * ac_scores[batch_idx, all_matches2[batch_idx][0], all_matches2[batch_idx][1]]
                validation_loss_list.append((item_loss.mean()).item())

            # calculate accuracy
            out = superglue.evaluate_matching(ac_scores, superglue.config['match_threshold'])

            correct_matches, superglue_matches = calculate_matches(gt_matches2, out)

            precision_list.extend([len(correct_match) / len(superglue_match) if len(superglue_match) != 0 else 0
                                   for correct_match, superglue_match in zip(correct_matches, superglue_matches)])
            recall_list.extend([len(correct_match) / gt_match.shape[1] if gt_match.shape[1] != 0 else 0
                                for correct_match, gt_match in zip(correct_matches, gt_matches2)])

            if index % 2 == 0:
                logger.info('')
                logger.info(' ID loss {} cycle loss {} triplet loss {} batch loss {} validation loss {} '.format(
                    id_loss_list[-1], cycle_loss_list[-1], triplet_loss_list[-1], batch_total_loss_list[-1], validation_loss_list[-1]))
                logger.info('val prec {} correct {} / {} '.format(precision_list[-1], len(correct_matches[-1]), len(superglue_matches[-1])))
                logger.info('val recall {} correct {} / {}'.format(recall_list[-1], len(correct_matches[-1]), gt_matches2[-1].shape[1]))

                # if debug_mode:

                #     scores_0 = out["matching_scores0"].detach().cpu().numpy().squeeze()
                #     draw_image_projection_match(image_kp_projected_to_warped, img, img_kp, wrp, verbose_mode)
                #     draw_superpoint_and_projected_match(image_kp_projected_to_warped, img, img_kp, warped_kp, gt_match, wrp,
                #                                         projected_to_warped_and_superpoints_dist, verbose_mode)
                #     draw_superglue_match(img, img_kp, superglue_match, warped_kp, wrp,
                #                          projected_to_warped_and_superpoints_dist, scores_0, verbose_mode)

            # if index > 4:
            #     break

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_triplet_loss = np.mean(triplet_loss_list)
        mean_id_loss = np.mean(id_loss_list)
        mean_cycle_loss = np.mean(cycle_loss_list)
        mean_val_loss = np.mean(validation_loss_list)
        mean_batch_loss = np.mean(batch_total_loss_list)

        writer.add_scalar('epoch/train/mean_triplet_loss', mean_triplet_loss, epoch_index)
        writer.add_scalar('epoch/train/mean_recall', mean_recall, epoch_index)
        writer.add_scalar('epoch/train/mean_batch_loss', mean_batch_loss, epoch_index)
        writer.add_scalar('epoch/train/mean_precision', mean_precision, epoch_index)
        writer.add_scalar('epoch/train/mean_id_loss', mean_id_loss, epoch_index)
        writer.add_scalar('epoch/train/mean_cycle_loss', mean_cycle_loss, epoch_index)
        writer.add_scalar('epoch/train/mean_val_loss', mean_val_loss, epoch_index)

        logger.info('')
        logger.info('epoch loss: ID loss {} cycle loss {} triplet loss {} batch loss {} validation loss {}'.format(
            mean_id_loss, mean_cycle_loss, mean_triplet_loss, mean_batch_loss, mean_val_loss))
        logger.info('epoch precision '.format(mean_precision, mean_recall))

        # superglue.eval()
        # mean_loss, mean_precision, mean_recall = calculate_validation_results(dataloader_validation, superglue)
        #
        #
        # writer.add_scalar('epoch/validation/mean_loss', mean_loss, epoch_index)
        # writer.add_scalar('epoch/validation/mean_recall', mean_recall, epoch_index)
        # writer.add_scalar('epoch/validation/mean_precision', mean_precision, epoch_index)
        # logger.info('validation epoch {} mean prec {} mean recall {} mean loss {}'.format(epoch_index, mean_precision, mean_recall,
        #                                                                        mean_loss))

        torch.save(superglue.state_dict(), experiment_folder_path / "model_{}.pth".format(epoch_index))


def calculate_cycle_loss(all_matches, data, gt_matches, superglue, max_keypoints):
    aba_scores = superglue.forward_cycle(data)
    gt_matches_cycle = []
    all_matches_cycle = []
    for all_match, gt_match in zip(all_matches, gt_matches):
        # take all matches that exist and expect it to be match to itself, rest go to dustbin
        gt_match_cycle = np.array([gt_match[0], gt_match[0]])
        gt_matches_cycle.append(gt_match_cycle)

        missing1 = np.setdiff1d(np.arange(max_keypoints), gt_match_cycle[0])
        missing2 = np.setdiff1d(np.arange(max_keypoints), gt_match_cycle[1])
        MN2 = np.concatenate(
            [missing1[np.newaxis, :],
             max_keypoints * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate(
            [max_keypoints * np.ones((1, len(missing2)), dtype=np.int64),
             missing2[np.newaxis, :]])
        all_matches_cycle.append(np.concatenate([gt_match, MN2, MN3], axis=1))

    item_loss_list = []
    loss_list_for_calculate_batch_loss = []
    for batch_idx in range(len(all_matches_cycle)):
        item_loss = -1.0 * aba_scores[batch_idx, all_matches_cycle[batch_idx][0], all_matches_cycle[batch_idx][1]]
        item_loss_list.append(item_loss.mean().item())
        loss_list_for_calculate_batch_loss.append(item_loss)
    cycle_loss = torch.cat(loss_list_for_calculate_batch_loss).mean()
    return item_loss_list, cycle_loss


def calc_id_loss(all_matches, data, superglue):

    aa_scores = superglue.forward_ID(data)
    all_matches_identity = [np.array([np.arange(0, 512), np.arange(0, 512)]) for i in range(len(all_matches))]
    # gt_matches_identity = [np.array([np.arange(0,512), np.arange(0,512)]) for i in range(len(all_matches))]

    item_loss_list = []
    loss_list_for_calculate_batch_loss = []
    for batch_idx in range(len(all_matches_identity)):
        item_loss = -1.0 * aa_scores[batch_idx, all_matches_identity[batch_idx][0], all_matches_identity[batch_idx][1]]
        item_loss_list.append(item_loss.mean().item())
        loss_list_for_calculate_batch_loss.append(item_loss)
    id_loss = torch.cat(loss_list_for_calculate_batch_loss).mean()
    return item_loss_list, id_loss


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


if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    train(train=True, debug_mode=False, verbose_mode=False)
