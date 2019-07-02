import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    

    def pdr_metric(self,class_id):
        """
        Precision and recall metric for each class
         class_id=2 for small obstacle [0-off road,1-on road]
        """
        truth_mask=self.gt_labels==class_id
        pred_mask=self.pred_labels==class_id

        true_positive=(truth_mask & pred_mask)
        true_positive=np.count_nonzero(true_positive==True)

        total=np.count_nonzero(truth_mask==True)
        pred=np.count_nonzero(pred_mask==True)

        recall=float(true_positive/total)
        precision=float(true_positive/pred)
        return recall,precision



    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        if len(self.gt_labels) == 0 and len(self.pred_labels) == 0:
            self.gt_labels=gt_image
            self.pred_labels=pre_image
        else:
            self.gt_labels=np.append(self.gt_labels,gt_image,axis=0)
            self.pred_labels=np.append(self.pred_labels,pre_image,axis=0)

        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.gt_labels=[] 
        self.pred_labels=[]




