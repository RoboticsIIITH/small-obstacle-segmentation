import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torchvision import transforms
from mypath import Path
from dataloaders import custom_transforms as tr

class SmallObs(data.Dataset):

	NUM_CLASSES = 3

	def __init__(self, args, file_paths, split='train'):

		self.file_paths = file_paths
		self.split = split
		self.args = args
		self.class_names = ['off_road','on_road','small_obstacle']

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):

		input_path = self.file_paths[index]
		temp=input_path.split('labels')
		img_path = temp[0] + 'image' + temp[1]
		rp_path = temp[0] + 'context_geometric' + temp[1].split('.')[0] + '.npy'
		# depth_path = temp[0] + 'depth' + temp[1]

		_img = np.array(Image.open(img_path))
		# _depth = np.array(Image.open(depth_path),dtype=np.float)
		# assert np.max(_depth) > 255. , "Found 8 bit depth, 16 bit depth is required"
		# _depth = _depth/256.																# Converts 16 bit uint depth to 0-255 float
		_target = np.asarray(Image.open(input_path))
		_region_prop = np.load(rp_path)
		# _region_prop = np.clip(_region_prop,0,1)
		# print(np.max(_region_prop),np.min(_region_prop),input_path)
		assert np.max(_region_prop) <= 1.0, "Incorrect region proposal input"

		"""Combine all small obstacle classes from 2 to 9"""
		_target = _target.flatten()
		_target = [x if x<2 else 2 for x in _target]
		_target = np.array(_target,dtype=np.float).reshape(_img.shape[0],_img.shape[1])
		sample={'image':_img,'rp':_region_prop,'label':_target}

		if self.split == 'train':
			return self.transform_tr(sample)

		elif self.split == 'val':
			return self.transform_val(sample)

		elif self.split == 'test':
			return self.transform_ts(sample)


	def transform_tr(self,sample):

		composed_transforms = transforms.Compose([
			tr.FixedCrop(x1=280,x2=1000,y1=50,y2=562),
			tr.RandomHorizontalFlip(),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()
			])
		return composed_transforms(sample)

	def transform_val(self,sample):
		composed_transforms = transforms.Compose([
			tr.FixedCrop(x1=280, x2=1000, y1=50, y2=562),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])

		return composed_transforms(sample)

	def transform_ts(self,sample):
		composed_transforms = transforms.Compose([
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