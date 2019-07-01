#from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd,small_obstacle
from dataloaders.datasets import small_obstacle
from torch.utils.data import DataLoader
from mypath import Path
import os
import random
random.seed(10)

def make_data_loader(args, **kwargs):
	"""
	if args.dataset == 'pascal':
		train_set = pascal.VOCSegmentation(args, split='train')
		val_set = pascal.VOCSegmentation(args, split='val')
		if args.use_sbd:
			sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
			train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		test_loader = None

		return train_loader, val_loader, test_loader, num_class

	elif args.dataset == 'cityscapes':
		train_set = cityscapes.CityscapesSegmentation(args, split='train')
		val_set = cityscapes.CityscapesSegmentation(args, split='val')
		test_set = cityscapes.CityscapesSegmentation(args, split='test')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader, test_loader, num_class

	elif args.dataset == 'coco':
		train_set = coco.COCOSegmentation(args, split='train')
		val_set = coco.COCOSegmentation(args, split='val')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		test_loader = None
		return train_loader, val_loader, test_loader, num_class
	"""
	if args.dataset == 'small_obstacle':

		images = []
		for folder in os.listdir(Path.db_root_dir(args.dataset)):
			path = os.path.join(Path.db_root_dir(args.dataset), folder, 'image')
			for image in os.listdir(path):
				images.append(path + '/' + image)

		random.shuffle(images)

		len_dataset = len(images)

		dataset_path = {}
		#dataset_path['train'] = images[:int(0.7 * len_dataset)]
		dataset_path['train'] = images
		#dataset_path['val'] = images[int(0.7 * len_dataset):int(0.9 * len_dataset)]
		dataset_path['val'] = images
		#dataset_path['test'] = images[int(0.9 * len_dataset):]
		dataset_path['test'] = images
		print("Dataset found ... Train Size: {}, Val Size: {}, Test Size: {}".format(len(dataset_path['train']),
																					 len(dataset_path['val']),
																					 len(dataset_path['test'])))

		train_set = small_obstacle.SmallObs(args,image_paths=dataset_path['train'],split='train')
		val_set = small_obstacle.SmallObs(args,image_paths=dataset_path['val'],split='val')
		test_set = small_obstacle.SmallObs(args,image_paths=dataset_path['test'],split='test')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,drop_last=True, **kwargs)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,drop_last=True, **kwargs)
		return train_loader, val_loader, test_loader, num_class

	else:
		raise NotImplementedError

