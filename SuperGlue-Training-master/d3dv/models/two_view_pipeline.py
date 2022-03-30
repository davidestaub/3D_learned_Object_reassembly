"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from .base_model import BaseModel
from . import get_model
from .utils.gt_matches import gt_matches_from_pose_depth, match_reward_matrix


class TwoViewPipeline(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'superpoint',
            'trainable': False,
        },
        'detector': {'name': None},
        'descriptor': {'name': None},
        'matcher': {'name': 'nearest_neighbor_matcher'},
        'filter': {'name': None},
        'solver': {'name': None},
        'ground_truth': {
            'from_pose_depth': False,
            'from_reward_matrix': False,
            'th_positive': 3,
            'th_negative': 5,
            'reward_positive': 1,
            'reward_negative': -0.25,
        },
    }
    required_data_keys = ['image0', 'image1']
    strict_conf = False  # need to pass new confs to children models

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(conf.extractor)
        else:
            if self.conf.detector.name:
                self.detector = get_model(conf.detector.name)(conf.detector)
            else:
                self.required_data_keys += ['keypoints0', 'keypoints1']
            if self.conf.descriptor.name:
                self.descriptor = get_model(conf.descriptor.name)(
                        conf.descriptor)
            else:
                self.required_data_keys += ['descriptors0', 'descriptors1']

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(conf.matcher)
        else:
            self.required_data_keys += ['matches0']

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(conf.filter)

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(conf.solver)

    def _forward(self, data):

        def process_siamese(data, i):
            data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
            if self.conf.extractor.name:
                pred_i = self.extractor(data_i)
            else:
                pred_i = {}
                if self.conf.detector.name:
                    pred_i = self.detector(data_i)
                else:
                    for k in ['keypoints', 'keypoint_scores', 'descriptors']:
                        if k in data_i:
                            pred_i[k] = data_i[k]
                if self.conf.descriptor.name:
                    pred_i = {
                        **pred_i, **self.descriptor({**data_i, **pred_i})}
            return pred_i

        pred0 = process_siamese(data, '0')
        pred1 = process_siamese(data, '1')
        pred = {**{k+'0': v for k, v in pred0.items()},
                **{k+'1': v for k, v in pred1.items()}}

        if self.conf.ground_truth.from_pose_depth:
            assignment, m0, m1 = gt_matches_from_pose_depth(
                pred['keypoints0'], pred['keypoints1'], **data,
                pos_th=self.conf.ground_truth.th_positive,
                neg_th=self.conf.ground_truth.th_negative)
            pred['gt_assignment'] = assignment
            pred['gt_matches0'], pred['gt_matches1'] = m0, m1
        elif self.conf.ground_truth.from_reward_matrix:
            reward, assignment, m0, m1 = match_reward_matrix(
                pred['keypoints0'], pred['keypoints1'],
                pos_score=self.conf.ground_truth.reward_positive,
                neg_score=self.conf.ground_truth.reward_negative,
                **data)
            pred['match_reward'] = reward
            pred['gt_assignment'] = assignment
            pred['gt_matches0'], pred['gt_matches1'] = m0, m1

        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}

        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}

        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        return pred

    def loss(self, pred, data):
        losses = {}
        total = 0
        for k in ['extractor', 'detector', 'descriptor', 'matcher', 'filter']:
            if self.conf[k].name:
                try:
                    losses_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                total = losses_['total'] + total
        return {**losses, 'total': total}

    def metrics(self, pred, data):
        metrics = {}
        for k in ['extractor', 'detector', 'descriptor', 'matcher', 'filter']:
            if self.conf[k].name:
                try:
                    metrics_ = getattr(self, k).metrics(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                metrics = {**metrics, **metrics_}
        return metrics
