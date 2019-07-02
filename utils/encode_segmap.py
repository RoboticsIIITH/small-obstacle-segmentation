import os
import cv2
from tqdm import tqdm
import  numpy as np
root_path='/scratch/ash/canyon_test2_run/'
folders=os.listdir(root_path)
tqdm_iter=tqdm(folders)
for f in tqdm_iter:
	seg_path = os.path.join(root_path, f + '/segmentation/')
	seg_files = os.listdir(seg_path)
	for files in seg_files:
		img = cv2.imread(os.path.join(seg_path, files))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		all_colors = np.unique(img)  # Dict: 107==Road 194 == of-road 241 == small obstacle
		if len(all_colors)>3:
			raise RuntimeError("More than 3 classes in segmentation")
		img[img == 194] = 0
		img[img == 107] = 1
		img[img == 241] = 2
		write_path = os.path.join(root_path, f + '/labels/')
		if not os.path.exists(write_path):
			os.mkdir(write_path)
		cv2.imwrite(os.path.join(write_path,files), img)





