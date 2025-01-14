'''
https://github.com/skylook/SuperGlue/blob/master/datasets/superpoint_dataset.py
'''

import sys

sys.path.append('./')

import numpy as np
import torch
import os
import cv2
import math
import datetime
import random
import glob

# from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

# from skimage import io, transform
# from skimage.color import rgb2gray

from models.superpoint import SuperPoint
from models.utils import frame2tensor, array2tensor

class SuperPointDataset(Dataset):

    def __init__(self, image_path, image_list=None, device='cpu', superpoint_config={}, image_size=(640, 480), max_keypoints=1024):

        print('Using SuperPoint dataset')

        self.DEBUG = False
        self.image_path = image_path
        self.device = device
        self.image_size = image_size
        self.max_keypoints = max_keypoints

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(image_path)
                if name.endswith('jpg') or name.endswith('png') ]
        
        self.image_names = [i for i in self.image_names if cv2.imread(os.path.join(self.image_path, i), cv2.IMREAD_GRAYSCALE) is not None]

        # Load SuperPoint model
        self.superpoint = SuperPoint(superpoint_config)
        self.superpoint.to(device)

    def __len__(self):
        return 50000 #len(self.image_names)

    def __getitem__(self, idx):
        idx = idx % len(self.image_names)
        # Read image
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size[::-1])
        height, width = image.shape[:2]
        min_size = min(height, width)

        # Transform image
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-min_size / 4, min_size / 4, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        image_warped = cv2.warpPerspective(image, M, (width, height))
        if self.DEBUG: print(f'Image size: {image.shape} -> {image_warped.shape}')

        with torch.no_grad():
            # Extract keypoints
            data = frame2tensor(image, self.device)
            pred0 = self.superpoint({ 'image': data })
            kps0 = pred0['keypoints'][0]
            desc0 = pred0['descriptors'][0]
            scores0 = pred0['scores'][0]
            
            # filter keypoints
            idxs = np.argsort(scores0.data.cpu().numpy())[::-1][:self.max_keypoints]
            scores0 = scores0[idxs.copy()]
            kps0 = kps0[idxs.copy(),:]
            desc0 = desc0[:,idxs.copy()]

            if self.DEBUG: print(f'Original keypoints: {kps0.shape}, descriptors: {desc0.shape}, scores: {scores0.shape}')

            # Transform keypoints
            kps1 = cv2.perspectiveTransform(kps0.cpu().numpy()[None], M)

            # Filter keypoints
            matches = [ [], [] ]
            kps1_filtered = []
            border = self.superpoint.config.get('remove_borders', 4)
            for i, k in enumerate(kps1.squeeze()):
                if k[0] < border or k[0] >= width - border: continue
                if k[1] < border or k[1] >= height - border: continue
                kps1_filtered.append(k)
                matches[0].append(i)
                matches[1].append(len(matches[1]))
            all_matches = [ torch.tensor(ms) for ms in matches ]
            kps1_filtered = array2tensor(np.array(kps1_filtered), self.device)

            # Compute descriptors & scores
            data_warped = frame2tensor(image_warped, self.device)
            desc1, scores1 = self.superpoint.computeDescriptorsAndScores({ 'image': data_warped, 'keypoints': kps1_filtered })
            if self.DEBUG: print(f'Transformed keypoints: {kps1_filtered.shape}, descriptor: {desc1[0].shape}, scores: {scores1[0].shape}')

        # Draw keypoints and matches
        if self.DEBUG:
            kps0cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze() ]
            kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps1_filtered.cpu().numpy().squeeze() ]
            matchescv = [ cv2.DMatch(k0, k1, 0) for k0,k1 in zip(matches[0], matches[1]) ]
            outimg = None
            outimg = cv2.drawMatches(image, kps0cv, image_warped, kps1cv, matchescv, outimg)
            cv2.imwrite('matches.jpg', outimg)
            outimg = cv2.drawKeypoints(image, kps0cv, outimg)
            cv2.imwrite('keypoints0.jpg', outimg)
            outimg = cv2.drawKeypoints(image_warped, kps1cv, outimg)
            cv2.imwrite('keypoints1.jpg', outimg)
        
        return {
            'keypoints0': kps0,
            'keypoints1': kps1_filtered[0],
            'descriptors0': desc0,
            'descriptors1': desc1[0],
            'scores0': scores0,
            'scores1': scores1[0],
            'image0': data.squeeze(0),
            'image1': data_warped.squeeze(0),
            'all_matches': all_matches,
            'file_name': self.image_names[idx],
            'homography' : M
        }

class SuperPointDatasetTest(Dataset):

    def __init__(self, image_path, image_list=None, device='cpu', superpoint_config={}):

        print('Using SuperPoint dataset')

        self.DEBUG = False
        self.image_path = image_path
        self.device = device

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(image_path)
                if name.endswith('jpg') or name.endswith('png') ]
        
        self.image_names = [i for i in self.image_names if cv2.imread(os.path.join(self.image_path, i), cv2.IMREAD_GRAYSCALE) is not None]

        # Load SuperPoint model
        self.superpoint = SuperPoint(superpoint_config)
        self.superpoint.to(device)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Read image
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        min_size = min(height, width)

        # Transform image
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-min_size / 4, min_size / 4, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        image_warped = cv2.warpPerspective(image, M, (width, height))
        if self.DEBUG: print(f'Image size: {image.shape} -> {image_warped.shape}')

        # Extract keypoints
        data = frame2tensor(image, self.device)
        pred0 = self.superpoint({ 'image': data })
        kps0 = pred0['keypoints'][0]
        desc0 = pred0['descriptors'][0]
        scores0 = pred0['scores'][0]
        if self.DEBUG: print(f'Original keypoints: {kps0.shape}, descriptors: {desc0.shape}, scores: {scores0.shape}')

        # Extract keypoints
        data_warped = frame2tensor(image_warped, self.device)
        pred1 = self.superpoint({ 'image': data_warped })
        kps1 = pred1['keypoints'][0]
        desc1 = pred1['descriptors'][0]
        scores1 = pred1['scores'][0]
        if self.DEBUG: print(f'Original keypoints: {kps1.shape}, descriptors: {desc1.shape}, scores: {scores1.shape}')


        # Draw keypoints and matches
        if self.DEBUG:
            kps0cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze() ]
            kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps1.cpu().numpy().squeeze() ]
            outimg = None
            outimg = cv2.drawKeypoints(image, kps0cv, outimg)
            cv2.imwrite('keypoints0.jpg', outimg)
            outimg = cv2.drawKeypoints(image_warped, kps1cv, outimg)
            cv2.imwrite('keypoints1.jpg', outimg)
        
        return {
            'keypoints0': kps0,
            'keypoints1': kps1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'image0': data.squeeze(0),
            'image1': data_warped.squeeze(0),
            'file_name': self.image_names[idx],
            'homography' : M
        }

class HPatches(Dataset):

    def __init__(self, typ="viewpoint", device='cuda', superpoint_config={}):

        print('Using SuperPoint dataset')

        self.DEBUG = False
        self.device = device
        self.typ = typ
        self.max_keypoints = 1024

        if self.typ == "illumination":
            self.folders = glob.glob("/nfs/WD_DS/Datasets/ImageMatching/hpatches-sequences-release/i_*")
        elif self.typ == "viewpoint":
            self.folders = glob.glob("/nfs/WD_DS/Datasets/ImageMatching/hpatches-sequences-release/v_*")
        else:
            assert 1==2, "not implemented"
        
        self.pairs = self.make_pairs(self.folders)

        # Load SuperPoint model
        self.superpoint = SuperPoint(superpoint_config)
        self.superpoint.to(device)
    
    def make_pairs(self, folders):
        '''
        1.ppm pairs with rest 2.ppm, 3.ppm .... 6.ppm
        return:
            [(1.ppm, 2.ppm, H_1_2),(1.ppm, 3.ppm, H_1_3) ... (1.ppm, 6.ppm, H_1_6)] 
        '''
        all_pairs = []
        for f in folders:
            im1 = os.path.join(f,'1.ppm')
            for j in range(2,7):
                H = np.loadtxt(os.path.join(f, f"H_1_{j}"))
                im2 = os.path.join(f,f"{j}.ppm")
                all_pairs.append([im1, im2, H])
        
        return all_pairs



    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, H = self.pairs[idx]
        
        # Read image
        image = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        min_size = min(height, width)

        # Extract keypoints
        data = frame2tensor(image, self.device)
        pred0 = self.superpoint({ 'image': data })
        kps0 = pred0['keypoints'][0]
        desc0 = pred0['descriptors'][0]
        scores0 = pred0['scores'][0]
        if self.DEBUG: print(f'Original keypoints: {kps0.shape}, descriptors: {desc0.shape}, scores: {scores0.shape}')

        # filter keypoints
        idxs = np.argsort(scores0.data.cpu().numpy())[::-1][:self.max_keypoints]
        scores0 = scores0[idxs.copy()]
        kps0 = kps0[idxs.copy(),:]
        desc0 = desc0[:,idxs.copy()]

        image_warped = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        # Extract keypoints
        data_warped = frame2tensor(image_warped, self.device)
        pred1 = self.superpoint({ 'image': data_warped })
        kps1 = pred1['keypoints'][0]
        desc1 = pred1['descriptors'][0]
        scores1 = pred1['scores'][0]
        if self.DEBUG: print(f'Original keypoints: {kps1.shape}, descriptors: {desc1.shape}, scores: {scores1.shape}')

        # filter keypoints
        idxs = np.argsort(scores1.data.cpu().numpy())[::-1][:self.max_keypoints]
        scores1 = scores1[idxs.copy()]
        kps1 = kps1[idxs.copy(),:]
        desc1 = desc1[:,idxs.copy()]


        # Draw keypoints and matches
        if self.DEBUG:
            kps0cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze() ]
            kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps1.cpu().numpy().squeeze() ]
            outimg = None
            outimg = cv2.drawKeypoints(image, kps0cv, outimg)
            cv2.imwrite('keypoints0.jpg', outimg)
            outimg = cv2.drawKeypoints(image_warped, kps1cv, outimg)
            cv2.imwrite('keypoints1.jpg', outimg)
        
        return {
            'keypoints0': kps0,
            'keypoints1': kps1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'image0': data.squeeze(0),
            'image1': data_warped.squeeze(0),
            'file_name': img2,
            'homography' : H
        }

def split_paris_oxford_dataset():
    
    def write_to_txt(my_list, name):
        with open(name, 'w') as f:
            for item in my_list:
                f.write("%s\n" % item)

    all_imnames = glob.glob("datasets/*/jpg/*.jpg")
    random.shuffle(all_imnames)

    train = all_imnames[:int(0.6*len(all_imnames))]
    val = all_imnames[int(0.6*len(all_imnames)) : int(0.8*len(all_imnames))]
    test = all_imnames[int(0.8*len(all_imnames)) : ]

    print(f"ALL : {len(all_imnames)}, TRAIN : {len(train)}, VAL : {len(val)}, TEST : {len(test)}")

    write_to_txt(train, 'datasets/train.txt')
    write_to_txt(val, 'datasets/val.txt')
    write_to_txt(test, 'datasets/test.txt')

if __name__ == '__main__':

    split_paris_oxford_dataset()

    config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
            }
    dataset = SuperPointDataset('datasets/roxford5k/jpg', device='cuda', superpoint_config=config)

    dataset.__getitem__(0)