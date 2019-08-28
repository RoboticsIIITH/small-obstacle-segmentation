import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def get_idr(pred, target, class_value, threshold=0.4):

    """Returns Instance Detection Ratio (IDR)
    for a given class, where class_id = numeric label of that class in segmentation target img
    Threshold is defined as minimum ratio of pixels between prediction and target above
    which an instance is defined to have been detected
    """

    pred_mask = pred == class_value
    target_mask = target == class_value
    instance_id,instance_num = ndi.label(target_mask)               # Return number of instances of given class present in target image
    count = 0

    for id in range(1,instance_num+1):                              # Background is given instance id zero
        x,y = np.where(instance_id == id)
        detection_ratio = np.count_nonzero(pred_mask[x,y])/np.count_nonzero(target_mask[x,y])
        if detection_ratio >= threshold:
            count += 1

    idr = count/instance_num
    return idr


def pdr_metric(pred, truth, class_id):

    """
    Returns Precision and recall metric for a particular class
    where class_id = numeric label in segmentation target img
    """

    truth_mask = truth == class_id
    pred_mask = pred == class_id
    true_positive = truth_mask & pred_mask
    true_positive = np.count_nonzero(true_positive)
    total = np.count_nonzero(truth_mask)
    recall = float(true_positive / total)
    precision = float(true_positive / np.count_nonzero(pred_mask))
    return recall, precision


if __name__=="__main__":

    target_image = Image.open("target.png")
    target_image = target_image.convert('L')
    target_image = np.asarray(target_image)

    pred_image = Image.open("prediction.png")
    pred_image = pred_image.convert('L')
    pred_image = np.asarray(pred_image)

    class_values = np.unique(target_image)                      # 70 for our desired case
    print("Class_id in segmentation label",class_values)
    idr = get_idr(pred_image,target_image,70)
    precision,recall = pdr_metric(pred_image,target_image,70)
    print("IDR: {}, Precision: {}, Recall: {}".format(idr,precision,recall))