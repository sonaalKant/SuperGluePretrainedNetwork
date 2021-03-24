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

def visualize(data, eval_output_dir):
    image0, image1 = data['image0'].cpu().numpy()[0]*255., data['image1'].cpu().numpy()[0]*255.
    kpts0, kpts1 = data['keypoints0'].cpu().numpy()[0], data['keypoints1'].cpu().numpy()[0]
    matches, conf = data['matches0'].cpu().detach().numpy()[0], data['matching_scores0'].cpu().detach().numpy()[0]
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    viz_path = os.path.join(eval_output_dir , data['file_name'][0].split('/')[-1].replace(".ppm", ".jpg"))
    color = cm.jet(mconf)
    stem = data['file_name']
    text = []
    out = make_matching_plot(image0[0], image1[0], kpts0, kpts1, mkpts0, mkpts1, color, text, viz_path, stem, False, True, False, 'Matches')

def hpatches_test():
    dataset = HPatches(typ="viewpoint")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, drop_last=True)
    return dataloader

def artificial_test(filename):
    dataset = SuperPointDatasetTest("", image_list=image_list, device='cuda')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, drop_last=True)
    return dataloader

def validation(model, dataloader, save_folder=None):

    homography_thresh = np.arange(10)
    correctness = []
    est_H_mean_dist = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), ncols=50):
            data['is_train'] = False
            
            pred = model(data)
            data = {**pred, **data}

            if save_folder is not None:
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                visualize(data, save_folder)
                
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
            est_H_mean_dist.append(result['mean_dist'])
        
    correctness_ave = np.array(correctness).mean(axis=0)
    homography_auc = auc(correctness_ave, np.array(homography_thresh))
    return homography_auc, np.mean(est_H_mean_dist)

if __name__ == '__main__':

    superglue = SuperGlue(config={})
    # superglue = torch.load("exp/model_epoch_4_0.0.pth")
    superglue.load_state_dict(torch.load("models/weights/superglue_outdoor.pth"))
    superglue.cuda()

    dataloader = hpatches_test()

    homography_auc, mean_dist = validation(superglue, dataloader)
    print(homography_auc, mean_dist)

