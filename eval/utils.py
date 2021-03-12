"""Script for descriptor evaluation

Updated by You-Yi from https://github.com/eric-yyjau/image_denoising_matching
Date: 2020/08/05

"""

import numpy as np
import cv2
from os import path as osp
from glob import glob


def keep_shared_points(keypoint_map, H, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(keypoints, H, keypoint_map.shape)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)

def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False, shape=(240,320)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """

    # print("shape: ", shape)
    shape = data['image0'].shape[-2:][::-1]
    real_H = data['homography'][0]

    m_keypoints = data['keypoints0'].permute(1,0,2) # shape : [kpts, 1, 2]
    m_warped_keypoints = data['keypoints1'].permute(1,0,2)

    m_keypoints = m_keypoints.data.cpu().numpy()
    m_warped_keypoints = m_warped_keypoints.data.cpu().numpy()

    matches = data['matches0'][0].data.cpu().numpy()

    #  remove the unmatched points
    valid = matches > -1
    m_keypoints = m_keypoints[valid]
    m_warped_keypoints = m_warped_keypoints[matches[valid]]
    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints[:, :, :], m_warped_keypoints[:, :, :], cv2.RANSAC)
                                    
    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        # print("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        # print("corner: ", corners)
        real_warped_corners = np.dot(corners, np.transpose(real_H))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        # print("real_warped_corners: ", real_warped_corners)
        
        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        # print("warped_corners: ", warped_corners)
        
        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        correctness = mean_dist <= correctness_thresh
    
    return {'correctness': correctness,
            'homography': H,
            'mean_dist': mean_dist
            }
