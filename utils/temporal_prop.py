import numpy as np
import cv2
import torch
import scipy.ndimage as sp
methods = ["cv2.TM_CCOEFF_NORMED"]


def get_mask(inp, span=15):
    instance_id, instance_num = sp.label(inp)
    mask = np.zeros((inp.shape[0], inp.shape[1]))
    for i in range(instance_num):
        x, y = np.where(instance_id == i + 1)
        min_x = np.min(x) - span
        min_y = np.min(y) - span
        max_x = np.max(x) + span
        max_y = np.max(y) + span
        mask[min_x:max_x, min_y:max_y] = 1
    return mask


def get_crop_bounds(b_x,b_y,size,h,w):
    bound_left = b_x - size if b_x - size > 0 else 0
    bound_right = b_x + size if b_x - size < h else h
    bound_down = b_y - size if b_y - size > 0 else 0
    bound_up = b_y + size if b_y + size < w else w
    return (bound_left,bound_right),(bound_down,bound_up)


class TemporalContexts:

    def __init__(self,history_len=5):
        self.img_buffer = []
        self.region_buffer = []
        self.pred_buffer = []
        self.history_len = history_len

    def append_buffer(self,img,region,pred):
        img = img.squeeze()
        region = region.squeeze()
        pred = pred.squeeze()

        if len(self.pred_buffer) == self.history_len:
            self.img_buffer.pop(0)
            self.region_buffer.pop(0)
            self.pred_buffer.pop(0)

        image = np.transpose(img, axes=[1, 2, 0])
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.img_buffer.append(image)
        self.region_buffer.append(region)
        self.pred_buffer.append(pred)

    def temporal_prop(self,img,context):

        img = img.squeeze()
        context = context.squeeze()

        image = np.transpose(img, axes=[1, 2, 0])
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_height, img_width = image.shape[0], image.shape[1]

        for j in range(len(self.pred_buffer)):

            # Select contexts lying on road
            frame_region = self.region_buffer[j]
            frame_label = self.pred_buffer[j]
            pred_mask = (frame_label == 2)
            pred_mask = get_mask(pred_mask).astype(int)
            region_mask = frame_region != 0

            valid_region = (region_mask & pred_mask)
            region_id, num_region = sp.label(valid_region)
            frame_img = self.img_buffer[j]

            for k in range(1, num_region + 1):
                x, y = np.where(region_id == k)
                c_x, c_y = int(np.mean(x)), int(np.mean(y))
                (bound_left, bound_right), (bound_down, bound_up) = get_crop_bounds(c_x, c_y, 20, img_height,
                                                                                    img_width)
                template = frame_img[bound_left:bound_right, bound_down:bound_up]
                src_region = frame_region[bound_left:bound_right, bound_down:bound_up]

                h, w = template.shape[0], template.shape[1]
                method = eval(methods[0])
                (left_margin, right_margin), (down_margin, up_margin) = get_crop_bounds(c_x, c_y, 150, img_height,
                                                                                        img_width)
                dest_template = image[left_margin:right_margin, down_margin:up_margin]

                # Apply template Matching
                try:
                    res = cv2.matchTemplate(dest_template, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                except:
                    print("Template match error")
                    continue

                if max_val >= 0.90:
                    center_point = (int((top_left[0] + bottom_right[0]) / 2) + down_margin,
                                    int((top_left[1] + bottom_right[1]) / 2) + left_margin)
                    x_0, y_0 = center_point[1], center_point[0]
                    left_corner = [0, 0]
                    right_corner = [0, 0]
                    left_corner[0] = top_left[0] + down_margin
                    left_corner[1] = top_left[1] + left_margin
                    right_corner[0] = bottom_right[0] + down_margin
                    right_corner[1] = bottom_right[1] + left_margin

                    # If weak or no context region is there, make context
                    if context[x_0, y_0] < 0.5:
                        context[left_corner[1]:right_corner[1],left_corner[0]:right_corner[0]] += src_region

        context = np.clip(context, 0, 1)
        context = torch.from_numpy(context).float()
        return context