import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import numpy as np
import pyquaternion
import cv2
import collections
import matplotlib as mpl

from .datasets import get_dataset
from .models import get_model
from .geometry.utils import T_to_E, sym_epipolar_distance
from .geometry.viz_2d import plot_images, plot_matches, cm_RdGn
from .utils.tools import set_seed
from .utils.tensor import batch_to_device, map_tensor
from .utils.experiments import load_experiment

try:
    import pycolmap
except ImportError:
    logging.warning('Could not find pycolmap')
try:
    import pyransac
except ImportError:
    logging.warning('Could not find pyransac')


default_eval_conf = {
    'seed': 0,
    'solver': 'cv2-RANSAC-E',
    'RANSAC_reproj_thresh': 1,
    'PR_reproj_thresh': 5e-4,
    'AUC_thresholds': [5, 10, 20],
    'split': 'test',
    'experiment': None,
    'max_length': None,
    'num_viz': 0,
}
default_eval_conf = OmegaConf.create(default_eval_conf)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_R = angle_error_mat(R, R_gt)
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    return error_R, error_t


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


def normalize_keypoints_intrinsics(kps, K):
    return (kps - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]


def estimate_relative_pose(kpts0, kpts1, K0, K1, thresh, solver, im0, im1):
    if (solver[-1] == 'E' and len(kpts0) < 5) or (len(kpts0) < 7):
        return None

    if solver == 'pycolmap-E':
        h0, w0 = im0.shape[:2]
        h1, w1 = im1.shape[:2]
        cd0 = {
            'model': 'SIMPLE_PINHOLE', 'width': w0, 'height': h0,
            'params': [K0[0, 0], K0[0, 2], K0[1, 2]]}
        cd1 = {
            'model': 'SIMPLE_PINHOLE', 'width': w1, 'height': h1,
            'params': [K1[0, 0], K1[0, 2], K1[1, 2]]}
        ret = pycolmap.essential_matrix_estimation(
            kpts0, kpts1, cd0, cd1, thresh)
        R = pyquaternion.Quaternion(ret['qvec']).rotation_matrix
        result = (R, ret['tvec'], np.array(ret['inliers']))
        return result if ret['success'] else None

    elif solver == 'pydegensac-F':
        F, mask = pyransac.findFundamentalMatrix(
            kpts0.astype(np.float64), kpts1.astype(np.float64), thresh)
        mask = mask.astype(np.uint8)
        E = K1.T @ F @ K0

    elif solver == 'cv2-RANSAC-E':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        norm_thresh = thresh / f_mean
        kpts0 = normalize_keypoints_intrinsics(kpts0, K0)
        kpts1 = normalize_keypoints_intrinsics(kpts1, K1)
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=0.99999,
            method=cv2.RANSAC)

    else:
        raise NotImplementedError(solver)

    assert E is not None
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


@torch.no_grad()
def evaluate(conf, model=None, dataloader=None, device='cuda'):
    conf.eval = OmegaConf.merge(default_eval_conf, conf.eval)
    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries

    if dataloader is None:
        assert 'batch_size' not in conf.data
        dataset = get_dataset(conf.data.name)(conf.data)
        dataloader = dataset.get_data_loader(conf.eval.split, shuffle=True)

    if model is None:
        if conf.eval.experiment is not None:
            logging.info(f'Loading experiment {conf.eval.experiment}')
            model = load_experiment(conf.eval.experiment, conf.model)
        else:
            model = get_model(conf.model.name)(conf.model).eval()
        model = model.to(device)

    set_seed(conf.eval.seed)
    results = collections.defaultdict(list)
    i = 0

    for data in tqdm(dataloader):
        data_ = batch_to_device(data, device)
        pred_ = model(data_)

        data = map_tensor(data, lambda x: x.cpu().numpy()[0])
        pred = map_tensor(pred_, lambda x: x.cpu().numpy()[0])
        del data_, pred_

        image0, image1 = data['image0'][0], data['image1'][0]
        kps0, kps1 = pred['keypoints0'], pred['keypoints1']
        matches = pred['matches0']
        valid = matches > -1
        mkps0 = kps0[valid]
        mkps1 = kps1[matches[valid]]

        K0, K1 = data['K0'], data['K1']
        T_0to1 = data['T_0to1']

        E_0to1 = T_to_E(torch.from_numpy(T_0to1))
        epi_errs = sym_epipolar_distance(
            torch.from_numpy(normalize_keypoints_intrinsics(mkps0, K0)),
            torch.from_numpy(normalize_keypoints_intrinsics(mkps1, K1)),
            E_0to1).numpy()
        correct = epi_errs < conf.eval.PR_reproj_thresh
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else np.nan
        matching_score = num_correct / len(kps0) if len(kps0) > 0 else np.nan

        ret = estimate_relative_pose(
            mkps0, mkps1, K0, K1, conf.eval.RANSAC_reproj_thresh,
            conf.eval.solver, image0, image1)
        if ret is None:
            err_t, err_R = np.inf, np.inf
            num_inliers = 0
        else:
            R, t, inliers = ret
            err_R, err_t = compute_pose_error(T_0to1, R, t)
            num_inliers = inliers.astype(int).sum()

        results['precision'] += [precision]
        results['matching_score'] += [matching_score]
        results['num_inliers'] += [num_inliers]
        results['error_R'] += [err_R]
        results['error_t'] += [err_t]

        if i < conf.eval.num_viz:
            plot_images([image0, image1])
            colors = cm_RdGn(1 - np.clip(epi_errs / 1e-3, 0, 1)).tolist()
            plot_matches(mkps0, mkps1, colors, a=0.5)
            e_R = 'FAIL' if np.isinf(err_R) else f'{err_R:.1f}°'
            e_t = 'FAIL' if np.isinf(err_t) else f'{err_t:.1f}°'
            text = [
                f'$\\Delta$R: {e_R}',
                f'$\\Delta$t: {e_t}',
                f'inliers: {np.sum(correct)}/{np.sum(matches>-1)}']
            txt_color = 'k' if image0[:100, :150].mean() > 0.7 else 'w'
            fig = mpl.pyplot.gcf()
            fig.text(
                0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
                fontsize=15, va='top', ha='left', color=txt_color)

        if conf.eval.max_length and i >= conf.eval.max_length:
            break
        i += 1

    pose_errors = np.maximum(results['error_t'], results['error_R'])
    aucs = pose_auc(pose_errors, conf.eval.AUC_thresholds)
    aucs = [100.*x for x in aucs]
    prec = 100.*np.nanmean(results['precision'])
    mscore = 100.*np.nanmean(results['matching_score'])

    return conf, aucs, prec, mscore
    # logging.info(f'Evaluation Results (mean over {i} pairs):')
    # logging.info('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    # logging.info('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            # aucs[0], aucs[1], aucs[2], prec, mscore))
