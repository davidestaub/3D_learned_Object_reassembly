import torch

from ...geometry.utils import to_homogeneous, from_homogeneous, T_to_E


def sample_depth(pts, depth):
    h, w = depth.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    batched = len(depth.shape) == 3
    if not batched:
        pts, depth = pts[None], depth[None]

    pts = (pts / pts.new_tensor([[w-1, h-1]]) * 2 - 1)[:, None]
    depth = torch.where(depth > 0, depth, depth.new_tensor(float('nan')))
    depth = depth[:, None]
    interp_lin = grid_sample(
            depth, pts, align_corners=True, mode='bilinear')[:, 0, 0]
    interp_nn = grid_sample(
            depth, pts, align_corners=True, mode='nearest')[:, 0, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = (~torch.isnan(interp)) & (interp > 0)
    if not batched:
        interp, valid = interp[0], valid[0]
    return interp, valid


def project(kpi, di, depthj, Ki, Kj, T_itoj, validi, rth=0.1):
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)
    consistent = (torch.abs(di_j - dj) / dj) < rth
    visible = validi & consistent & validj
    return kpi_j, visible


def sym_epipolar_distance_all(p0, p1, E, eps=1e-15):
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum('...mi,...ij,...nj->...nm', p1, E, p0).abs()
    E_p0 = torch.einsum('...ij,...nj->...ni', E, p0)
    Et_p1 = torch.einsum('...ij,...mi->...mj', E, p1)
    d0 = p1_E_p0 / (E_p0[..., None, 0]**2 + E_p0[..., None, 1]**2 + eps).sqrt()
    d1 = p1_E_p0 / (
            Et_p1[..., None, :, 0]**2 + Et_p1[..., None, :, 1]**2 + eps).sqrt()
    return (d0 + d1) / 2


@torch.no_grad()
def gt_matches_from_pose_depth(
        kp0, kp1, depth0, depth1, K0, K1, T_0to1, T_1to0,
        pos_th=3, neg_th=5, **kw):

    d0, valid0 = sample_depth(kp0, depth0)
    d1, valid1 = sample_depth(kp1, depth1)

    kp0_1, visible0 = project(kp0, d0, depth1, K0, K1, T_0to1, valid0)
    kp1_0, visible1 = project(kp1, d1, depth0, K1, K0, T_1to0, valid1)
    mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3))**2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3))**2, -1)
    dist = torch.max(dist0, dist1)
    inf = dist.new_tensor(float('inf'))
    dist = torch.where(mask_visible, dist, inf)

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    negative0 = (dist0.min(-1).values > neg_th**2) & valid0
    negative1 = (dist1.min(-2).values > neg_th**2) & valid1

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    unmatched, ignore = min0.new_tensor(-1), min0.new_tensor(-2)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    # Add some more unmatched points using epipolar geometry
    F = K1.inverse().transpose(-1, -2) @ T_to_E(T_0to1) @ K0.inverse()
    epi_dist = sym_epipolar_distance_all(kp0, kp1, F)
    mask_ignore = (m0.unsqueeze(-1) == ignore) & (m1.unsqueeze(-2) == ignore)
    epi_dist = torch.where(mask_ignore, epi_dist, inf)
    exclude0 = epi_dist.min(-1).values > neg_th
    exclude1 = epi_dist.min(-2).values > neg_th
    m0 = torch.where((~valid0) & exclude0, ignore.new_tensor(-3), m0)
    m1 = torch.where((~valid1) & exclude1, ignore.new_tensor(-3), m1)

    return positive, m0, m1


@torch.no_grad()
def match_reward_matrix(
        kp0, kp1, depth0, depth1, K0, K1, T_0to1, T_1to0,
        pos_th=3, neg_th=5,
        pos_score=1, neg_score=-0.25, plausible_score=0, **kw):

    d0, valid0 = sample_depth(kp0, depth0)
    d1, valid1 = sample_depth(kp1, depth1)

    kp0_1, visible0 = project(kp0, d0, depth1, K0, K1, T_0to1, valid0)
    kp1_0, visible1 = project(kp1, d1, depth0, K1, K0, T_1to0, valid1)
    # mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kp1.unsqueeze(-3))**2, -1)
    dist1 = torch.sum((kp0.unsqueeze(-2) - kp1_0.unsqueeze(-3))**2, -1)
    depth_dist = torch.max(dist0, dist1)

    F = K1.inverse().transpose(-1, -2) @ T_to_E(T_0to1) @ K0.inverse()
    epi_dist = sym_epipolar_distance_all(kp0, kp1, F)

    # Different from DISK, a plausible match should also have a low
    # reprojection error if depth is available for any of the two keypoints.
    # We use different thresholds for positive and negative to give some slack.
    has_depth_in_both = valid0.unsqueeze(-1) & valid1.unsqueeze(-2)
    has_depth_in_0 = valid0.unsqueeze(-1) & ~valid1.unsqueeze(-2)
    has_depth_in_1 = ~valid0.unsqueeze(-1) & valid1.unsqueeze(-2)
    negative = ((has_depth_in_both & (torch.min(dist0, dist1) > neg_th**2))
                | (has_depth_in_0 & (dist0 > neg_th**2))
                | (has_depth_in_1 & (dist1 > neg_th**2))
                | (epi_dist > neg_th))
    positive = has_depth_in_both & (depth_dist < pos_th**2)

    reward = epi_dist.new_tensor(plausible_score)
    reward = torch.where(positive, reward.new_tensor(pos_score), reward)
    reward = torch.where(negative, reward.new_tensor(neg_score), reward)

    inf = depth_dist.new_tensor(float('inf'))
    dist = torch.where(has_depth_in_both, depth_dist, inf)
    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices
    # ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    # ismin1 = ismin0.clone()
    # ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    # ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    # positive = positive & ismin0 & ismin1

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    # negative0 = (dist0.min(-1).values > neg_th**2) & valid0
    # negative1 = (dist1.min(-2).values > neg_th**2) & valid1
    negative0 = torch.all(negative, -1)
    negative1 = torch.all(negative, -2)
    unmatched, ignore = min0.new_tensor(-1), min0.new_tensor(-2)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    return reward, positive, m0, m1
