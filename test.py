from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.utils import *
from dataset import *
from eval.utils import *
from tqdm import tqdm
from sklearn.metrics import auc

if __name__ == '__main__':

    superglue = SuperGlue(config={})
    superglue.load_state_dict(torch.load("models/weights/superglue_outdoor.pth"))
    superglue.cuda()

    dataset = SuperPointDatasetTest("", image_list='datasets/test.txt', device='cuda')

    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, drop_last=True)
    
    homography_thresh = [1,3,5,10,20,50]
    correctness = []
    est_H_mean_dist = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), ncols=50):
            pred = superglue(data)
            data = {**pred, **data}
                
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
        
    correctness_ave = np.array(correctness).mean(axis=0)
    homogrpahy_auc = auc(correctness_ave, np.array(homography_thresh))
    print(homogrpahy_auc)
