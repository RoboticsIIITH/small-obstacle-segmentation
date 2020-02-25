import numpy as np
import scipy.ndimage as ndi


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.idr_count = 0

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
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def pdr_metric(self, class_id):
        """
        Precision and recall metric for each class
         class_id=2 for small obstacle [0-off road,1-on road]
        """

        # TODO:Memory error occured with batched implementation.Find a way to remove this loop later.

        recall_list = []
        precision_list = []

        for i in range(len(self.gt_labels)):
            truth_mask = self.gt_labels[i] == class_id
            pred_mask = self.pred_labels[i] == class_id

            true_positive = (truth_mask & pred_mask)
            true_positive = np.count_nonzero(true_positive == True)
            total = np.count_nonzero(truth_mask == True)
            pred = np.count_nonzero(pred_mask == True)
            if total:
                recall = float(true_positive / total)
                recall_list.append(recall)
                if pred == 0:
                    precision = 0.0
                else:
                    precision = float(true_positive / pred)
                precision_list.append(precision)

        return np.mean(recall_list), np.mean(precision_list)

    def get_idr(self, class_value, threshold=0.4):

        """Returns Instance Detection Ratio (IDR)
        for a given class, where class_id = numeric label of that class in segmentation target img
        Threshold is defined as minimum ratio of pixels between prediction and target above
        which an instance is defined to have been detected
        """
        pred = self.pred_labels
        target = self.gt_labels
        idr = []
        idr_count = 0
        for num in range(target.shape[0]):
            pred_mask = pred[num] == class_value
            target_mask = target[num] == class_value
            instance_id, instance_num = ndi.label(
                target_mask)  # Return number of instances of given class present in target image
            count = 0
            if instance_num == 0:
                idr.append(0.0)
            else:
                for id in range(1, instance_num + 1):  # Background is given instance id zero
                    x, y = np.where(instance_id == id)
                    detection_ratio = np.count_nonzero(pred_mask[x, y]) / np.count_nonzero(target_mask[x, y])
                    if detection_ratio >= threshold:
                        count += 1

                idr.append(float(count / instance_num))
                idr_count += 1

        idr = np.sum(idr) / idr_count
        return idr

    def get_false_idr(self,class_value):
        pred = self.pred_labels
        target = self.gt_labels
        false_idr = []
        false_idr_count = 0
        for num in range(target.shape[0]):
            pred_mask = pred[num] == class_value
            obstacle_mask = (target[num] == class_value).astype(int)
            road_mask = target[num] >= 1
            pred_mask = (pred_mask & road_mask).astype(int)         # Filter predictions lying on road
            instance_id, instance_num = ndi.label(pred_mask)        # Return predicted instances on road
            count = 0
            if instance_num == 0:
                false_idr.append(0.0)
            else:
                for id in range(1, instance_num + 1):   # Background is given instance id zero
                    x, y = np.where(instance_id == id)
                    is_false_detection = np.count_nonzero(pred_mask[x, y] & obstacle_mask[x,y])
                    if is_false_detection == 0:         # No overlap between prediction and label: Is a False detection
                        count += 1

                false_idr.append(float(count / instance_num))
                false_idr_count += 1

        false_idr_batch = np.sum(false_idr) / false_idr_count
        return false_idr_batch

    def get_instance_iou(self,threshold,class_value=2):
        pred = self.pred_labels
        target = self.gt_labels
        instance_iou=[]
        valid_frame_count = 0
        for num in range(target.shape[0]):
            true_positive = 0
            false_negative = 0
            false_positive = 0

            pred_mask = pred[num] == class_value
            target_mask = target[num] == class_value
            instance_id, instance_num = ndi.label(target_mask)      # Return number of instances of given class in target

            if instance_num == 0:
                instance_iou.append(0.0)
                continue
            else:
                for id in range(1, instance_num + 1):               # Background is given instance id zero
                    x, y = np.where(instance_id == id)
                    detection_ratio = np.count_nonzero(pred_mask[x, y]) / np.count_nonzero(target_mask[x, y])
                    if detection_ratio >= threshold:
                        true_positive += 1
                    else:
                        false_negative += 1

            road_mask = target[num] >= 1
            pred_on_road = (pred_mask & road_mask)
            instance_id, instance_num = ndi.label(pred_on_road)

            if instance_num == 0:
                false_positive = 0
            else:
                for id in range(1, instance_num + 1):
                    x, y = np.where(instance_id == id)
                    is_false_detection = np.count_nonzero(pred_on_road[x, y] & target_mask[x,y])
                    if is_false_detection == 0:         # No overlap between prediction and label: Is a False detection
                        false_positive += 1

            iIOU = true_positive / (true_positive + false_positive + false_negative)
            instance_iou.append(iIOU)
            valid_frame_count += 1

        iIOU_batch = float(np.sum(instance_iou)/valid_frame_count)
        return iIOU_batch


    def add_batch(self, gt_image, pre_image, *args):
        assert gt_image.shape == pre_image.shape
        if len(self.gt_labels) == 0 and len(self.pred_labels) == 0:
            self.gt_labels = gt_image
            self.pred_labels = pre_image
        else:
            self.gt_labels = np.append(self.gt_labels, gt_image, axis=0)
            self.pred_labels = np.append(self.pred_labels, pre_image, axis=0)

        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.gt_labels = []
        self.pred_labels = []
        self.idr_count = 0