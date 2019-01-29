# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

from losses import l1_loss_instance
import numpy as np

class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0.0
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes**2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum()) 
            self.num_updates += predictions.shape[0]
    
        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti!=250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum( np.abs(gti[mask] - _pred.astype(np.int32)[mask]) ) 
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        
    def get_result(self):
        if self._metric_type == 'ACC':
            return {'acc': self.accuracy/self.num_updates}
        if self._metric_type == 'L1':
            return {'l1': self.l1/self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) 
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc':acc_cls, 'mIOU': mean_iou}


def get_metrics(params):
    met = {}
    if 'mnist' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'ACC')
    if 'cityscapes' in params['dataset']:
        if 'S' in params['tasks']:
            met['S'] = RunningMetric(metric_type = 'IOU', n_classes=19)
        if 'I' in params['tasks']:
            met['I'] = RunningMetric(metric_type = 'L1')
        if 'D' in params['tasks']:
            met['D'] = RunningMetric(metric_type = 'L1')
    if 'celeba' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'ACC')
    return met