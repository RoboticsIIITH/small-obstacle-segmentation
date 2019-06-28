import os
import sys
sys.path.append('/home/ash/deeplab/')
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torchvision import transforms
from mypath import Path
from dataloaders import custom_transforms as tr

class SmallObs(data.Dataset):

	NUM_CLASSES = 3

	def __init__(self, args, image_paths, split='train'):

		self.image_paths=image_paths
		self.split = split
		self.args = args
		"""
		self.images_base = os.path.join(self.root,self.split,'image')
		self.annotations_base = os.path.join(self.root,self.split,'segmentation')
		self.class_names = ['off_road','on_road','small_obstacle']
		self.input_files=sorted(os.listdir(self.images_base))

		if len(self.input_files)==0:
			raise Exception("No files found in directory: {}".format(self.images_base))
		else:
			print("Found %d images" % (len(self.input_files)))
		"""

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):

		input_path = self.image_paths[index]
		temp=input_path.split('image')
		target_path = temp[0] + 'labels' + temp[1]
		_img = np.asarray(Image.open(input_path))[256:768,:,:3]
		_target = np.asarray(Image.open(target_path))[256:768,:]
		_img=Image.fromarray(_img)
		_target=Image.fromarray(_target)
		sample={'image':_img,'label':_target}

		if self.split == 'train':
			return self.transform_tr(sample)

		elif self.split == 'val':
			return self.transform_val(sample)

		elif self.split == 'test':
			return self.transform_ts(sample)


	def transform_tr(self,sample):

		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),
			tr.RandomCrop(crop_size=(512,512)),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()
			])
		return composed_transforms(sample)

	def transform_val(self,sample):
		composed_transforms = transforms.Compose([
			tr.RandomHorizontalFlip(),
			tr.RandomCrop(crop_size=(512,512)),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])

		return composed_transforms(sample)

	def transform_ts(self,sample):
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])

		return composed_transforms(sample)


"""
if __name__ == '__main__':
	#from dataloaders.utils import decode_segmap
	from torch.utils.data import DataLoader
	import matplotlib.pyplot as plt
	import argparse

	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	args.base_size = 512
	args.crop_size = 512

	cityscapes_train = SmallObs(args,split='train')
	trainloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)
	images,labels=next(iter(trainloader))
	print(images.shape,labels.shape,type(labels))
"""