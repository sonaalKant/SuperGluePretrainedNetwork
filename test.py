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

if __name__ == '__main__':

    superglue = SuperGlue(config={})
    superglue.load_state_dict(torch.load("models/weights/superglue_outdoor.pth"))
    superglue.cuda()

    dataset = SuperPointDataset("", image_list='datasets/test.txt', device='cuda')

    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, drop_last=True)
    
    homography = True
    correctness = []
    est_H_mean_dist = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    rep_thd = 3
    compute_map = True
    verbose = True
    top_K = 1000
    
    for i, data in enumerate(dataloader):
        pred = superglue(data)
        data = {**pred, **data}

        '''
        if args.repeatibility:
            rep, local_err = compute_repeatability(data, keep_k_points=top_K, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            print("repeatability: %.2f"%(rep))
            if local_err > 0:
                localization_err.append(local_err)
                print('local_err: ', local_err)
            if args.outputImg:
                # img = to3dim(image)
                img = image
                pts = data['prob']
                img1 = draw_keypoints(img*255, pts.transpose())

                # img = to3dim(warped_image)
                img = warped_image
                pts = data['warped_prob']
                img2 = draw_keypoints(img*255, pts.transpose())

                plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
                plt.title("rep: " + str(repeatability[-1]))
                plt.tight_layout()
                
                plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')
                pass
        '''

        if homography:
            # estimate result
            ##### check
            homography_thresh = [1,3,5,10,20,50]
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
            # est_H_mean_dist.append(result['mean_dist'])
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                import torch
                """
                input:
                    pnts: numpy
                    homography: numpy
                output:
                    warped_pnts: numpy
                """
                from utils.utils import warp_points
                from utils.utils import filter_points
                pnts = torch.tensor(pnts).long()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                          homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
                return warped_pnts.numpy()

            from numpy.linalg import inv
            H, W = image.shape
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            score = (result['inliers'].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            print("m. score: ", score)
            mscore.append(score)
            # compute map
            if compute_map:
                def getMatches(data):
                    from models.model_wrap import PointTracker

                    desc = data['desc']
                    warped_desc = data['warped_desc']

                    nn_thresh = 1.2
                    print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T

                    # mAP
                    # matches = data['matches']
                    print("matches: ", matches.shape)
                    print("mscores: ", mscores.shape)
                    print("mscore max: ", mscores.max(axis=0))
                    print("mscore min: ", mscores.min(axis=0))

                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    from evaluations.detector_evaluation import warp_keypoints
                    # warp points 
                    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                                            ord=None, axis=1)
                    inliers = norm < epi
                    if verbose:
                        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                                          ", percentage: ", inliers.sum() / inliers.shape[0])

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    import cv2
                    # count inliers: use opencv homography estimation
                    # Estimate the homography between the matches using RANSAC
                    H, inliers = cv2.findHomography(matches[:, [0, 1]],
                                                    matches[:, [2, 3]],
                                                    cv2.RANSAC)
                    inliers = inliers.flatten()
                    print("Total matches: ", inliers.shape[0], 
                          ", inliers: ", inliers.sum(),
                          ", percentage: ", inliers.sum() / inliers.shape[0])
                    return inliers
            
            
                def computeAP(m_test, m_score):
                    from sklearn.metrics import average_precision_score

                    average_precision = average_precision_score(m_test, m_score)
                    print('Average precision-recall score: {0:0.2f}'.format(
                        average_precision))
                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr
                
                if args.sift:
                    assert result is not None
                    matches, mscores = result['matches'], result['mscores']
                else:
                    matches, mscores = getMatches(data)
                
                real_H = data['homography']
                if inliers_method == 'gt':
                    # use ground truth homography
                    print("use ground truth homography for inliers")
                    inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
                else:
                    # use opencv estimation as inliers
                    print("use opencv estimation for inliers")
                    inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)
                    
                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:,2])
        
                if inliers.shape[0] > 0 and inliers.sum()>0:
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0
                
                mAP.append(ap)


    '''        
    if args.repeatibility:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    '''
    
    if homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)
        # print(f"mean est H dist: {est_H_mean_dist.mean()}")
        mscore_m = np.array(mscore).mean(axis=0)
        print("matching score", mscore_m)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")



    # # save to files
    # with open(save_file, "a") as myfile:
    #     myfile.write("path: " + path + '\n')
    #     myfile.write("output Images: " + str(args.outputImg) + '\n')
    #     if args.repeatibility:
    #         myfile.write("repeatability threshold: " + str(rep_thd) + '\n')
    #         myfile.write("repeatability: " + str(repeatability_ave) + '\n')
    #         myfile.write("localization error: " + str(localization_err_m) + '\n')
    #     if args.homography:
    #         myfile.write("Homography estimation: " + '\n')
    #         myfile.write("Homography threshold: " + str(homography_thresh) + '\n')
    #         myfile.write("Average correctness: " + str(correctness_ave) + '\n')

    #         # myfile.write("mean est H dist: " + str(est_H_mean_dist.mean()) + '\n')

    #         if compute_map:
    #             myfile.write("nn mean AP: " + str(mAP_m) + '\n')
    #         myfile.write("matching score: " + str(mscore_m) + '\n')



    #     if verbose:
    #         myfile.write("====== details =====" + '\n')
    #         for i in range(len(files)):

    #             myfile.write("file: " + files[i])
    #             if args.repeatibility:
    #                 myfile.write("; rep: " + str(repeatability[i]))
    #             if args.homography:
    #                 myfile.write("; correct: " + str(correctness[i]))
    #                 # matching
    #                 myfile.write("; mscore: " + str(mscore[i]))
    #                 if compute_map:
    #                     myfile.write(":, mean AP: " + str(mAP[i]))
    #             myfile.write('\n')
    #         myfile.write("======== end ========" + '\n')

    # dict_of_lists = {
    #     'repeatability': repeatability,
    #     'localization_err': localization_err,
    #     'correctness': np.array(correctness),
    #     'homography_thresh': homography_thresh,
    #     'mscore': mscore,
    #     'mAP': np.array(mAP),
    #     # 'est_H_mean_dist': est_H_mean_dist
    # }

    # filename = f'{save_file[:-4]}.npz'
    # logging.info(f"save file: {filename}")
    # np.savez(
    #     filename,
    #     **dict_of_lists,
    # )

