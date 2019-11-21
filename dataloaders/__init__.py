from dataloaders.datasets import small_obstacle
from torch.utils.data import DataLoader
from mypath import Path
import os
import random
random.seed(10)

def make_data_loader(args, **kwargs):

	if args.dataset == 'small_obstacle':

		train_path = os.path.join(Path.db_root_dir(args.dataset),"train")
		val_path = os.path.join(Path.db_root_dir(args.dataset),"val")
		test_path = os.path.join(Path.db_root_dir(args.dataset),"test")

		files_train = []
		files_val = []
		files_test = []

		for folder in os.listdir(train_path):
			path = os.path.join(train_path, folder,"labels")
			for file in sorted(os.listdir(path)):
				files_train.append(path + '/' + file)

		for folder in os.listdir(val_path):
		# for folder in ["stadium_3"]:
			path = os.path.join(val_path, folder,"labels")
			for file in sorted(os.listdir(path)):
				files_val.append(path + '/' + file)

		for folder in os.listdir(test_path):
			path = os.path.join(test_path, folder,"labels")
			for file in sorted(os.listdir(path)):
				files_test.append(path + '/' + file)


		dataset_path = {}
		dataset_path['train'] = files_train
		dataset_path['val'] = files_val
		dataset_path['test'] = files_test

		print("Dataset found ... Train Size: {}, Val Size: {}, Test Size: {}".format(len(dataset_path['train']),
																					 len(dataset_path['val']),
																					 len(dataset_path['test'])))

		train_set = small_obstacle.SmallObs(args,file_paths = dataset_path['train'],split='train')
		val_set = small_obstacle.SmallObs(args,file_paths = dataset_path['val'],split='val')
		test_set = small_obstacle.SmallObs(args,file_paths = dataset_path['test'],split='test')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=64, shuffle=True, **kwargs)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		return train_loader, val_loader, test_loader, num_class

	else:
		raise NotImplementedError