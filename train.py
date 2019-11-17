import argparse
import os
import numpy as np
from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weights_batch
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
	def __init__(self, args):
		self.args = args

		# Define Saver
		self.saver = Saver(args)
		self.saver.save_experiment_config()
		# Define Tensorboard Summary
		self.summary = TensorboardSummary(self.saver.experiment_dir)
		self.writer = self.summary.create_summary()

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

		# Define network
		model = DeepLab(num_classes=self.nclass,
						backbone=args.backbone,
						output_stride=args.out_stride,
						sync_bn=args.sync_bn,
						freeze_bn=args.freeze_bn)

		train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
						{'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

		# Define Optimizer
		optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
									weight_decay=args.weight_decay, nesterov=args.nesterov)

		# Define Criterion

		self.criterion = SegmentationLosses(cuda=args.cuda)
		self.model, self.optimizer = model, optimizer

		# Define Evaluator
		self.evaluator = Evaluator(self.nclass)
		# Define lr scheduler
		self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
											args.epochs, len(self.train_loader))

		# Using cuda
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			patch_replication_callback(self.model)
			self.model = self.model.cuda()

		# Resuming checkpoint
		self.best_pred = 0.0
		if args.resume is not None:
			if not os.path.isfile(args.resume):
				raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			if args.cuda:
				self.model.module.load_state_dict(checkpoint['state_dict'])
			else:
				self.model.load_state_dict(checkpoint['state_dict'])
			if not args.ft:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.best_pred = checkpoint['best_pred']
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

		# Clear start epoch if fine-tuning or in validation/test mode
		if args.ft or args.mode == "val" or args.mode == "test":
			args.start_epoch = 0
			self.best_pred = 0.0


	def training(self, epoch):
		train_loss = 0.0
		self.model.train()
		tbar = tqdm(self.train_loader)
		num_img_tr = len(self.train_loader)

		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label']
			if self.args.cuda:
				image, target = image.cuda(), target.cuda()

			self.scheduler(self.optimizer, i, epoch, self.best_pred)
			self.optimizer.zero_grad()
			output = self.model(image)
			loss = self.criterion.CrossEntropyLoss(output,target,weight=torch.from_numpy(calculate_weights_batch(sample,self.nclass).astype(np.float32)))
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
			self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

			pred = output.clone().data.cpu()
			pred_softmax = F.softmax(pred, dim=1).numpy()
			pred = np.argmax(pred.numpy(), axis=1)

			# Plot prediction every 20th iter
			if i % (num_img_tr // 20) == 0:
				global_step = i + num_img_tr * epoch
				self.summary.vis_grid(self.writer, self.args.dataset, image.clone().data.cpu().numpy()[0],
									target.clone().data.cpu().numpy()[0],pred[0],
									pred_softmax[0], global_step, split="Train")

		self.writer.add_scalar('train/total_loss_epoch', train_loss/num_img_tr, epoch)
		print('Loss: {}'.format(train_loss / num_img_tr))

		if self.args.no_val or self.args.save_all:
			# save checkpoint every epoch
			is_best = False
			self.saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,
			}, is_best, filename='checkpoint_' + str(epoch + 1) + '_.pth.tar')


	def validation(self, epoch):

		if self.args.mode=="train" or self.args.mode=="val":
			loader=self.val_loader
		elif self.args.mode=="test":
			loader=self.test_loader

		self.model.eval()
		self.evaluator.reset()
		tbar = tqdm(loader, desc='\r')

		test_loss = 0.0
		idr_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

		num_itr=len(loader)

		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label']
			if self.args.cuda:
				image, target = image.cuda(), target.cuda()
			with torch.no_grad():
				output = self.model(image)
			# loss = self.criterion.CrossEntropyLoss(output,target,weight=torch.from_numpy(calculate_weights_batch(sample,self.nclass).astype(np.float32)))
			# test_loss += loss.item()
			tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))


			pred = output.clone().data.cpu()
			pred_softmax = F.softmax(pred, dim=1).numpy()
			pred = np.argmax(pred.numpy(), axis=1)
			target = target.clone().data.cpu().numpy()
			image = image.clone().data.cpu().numpy()

			# Add batch sample into evaluator
			self.evaluator.add_batch(target, pred)
			global_step = i + num_itr * epoch
			self.summary.vis_grid(self.writer, self.args.dataset, image[0], target[0], pred[0],pred_softmax[0], global_step, split="Validation")

		# Fast test during the training
		Acc = self.evaluator.Pixel_Accuracy()
		Acc_class = self.evaluator.Pixel_Accuracy_Class()
		mIoU = self.evaluator.Mean_Intersection_over_Union()
		FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
		recall,precision=self.evaluator.pdr_metric(class_id=2)
		idr_avg = np.array([self.evaluator.get_idr(class_value=2, threshold=value) for value in idr_thresholds])

		self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
		self.writer.add_scalar('val/mIoU', mIoU, epoch)
		self.writer.add_scalar('val/Acc', Acc, epoch)
		self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
		self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
		self.writer.add_scalar('Recall/per_epoch',recall,epoch)
		self.writer.add_scalar('Precision/per_epoch',precision,epoch)
		self.writer.add_scalar('IDR/per_epoch(0.20)', idr_avg[0], epoch)
		self.writer.add_scalar('IDR/avg_epoch', np.mean(idr_avg), epoch)
		self.writer.add_histogram('Prediction_hist', self.evaluator.pred_labels[self.evaluator.gt_labels == 2], epoch)


		print('Validation:')
		print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
		print('Loss: %.3f' % test_loss)
		print('Recall/PDR:{}'.format(recall))
		print('Precision:{}'.format(precision))
		print('IDR:{}'.format(np.mean(idr_avg)))

		if self.args.mode == "train":
			new_pred = mIoU
			if new_pred > self.best_pred:
				is_best = True
				self.best_pred = new_pred
				self.saver.save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': self.model.module.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'best_pred': self.best_pred,
				}, is_best)

		else:
			pass



def main():
	parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
	parser.add_argument('--backbone', type=str, default='drn',
						choices=['resnet', 'xception', 'drn', 'mobilenet'],
						help='backbone name (default: drn)')
	parser.add_argument('--out-stride', type=int, default=16,
						help='network output stride (default: 8)')
	parser.add_argument('--dataset', type=str, default='small_obstacle',
						choices=['pascal', 'coco', 'cityscapes'],
						help='dataset name (default: pascal)')
	parser.add_argument('--use-sbd', action='store_true', default=False,
						help='whether to use SBD dataset (default: True)')
	parser.add_argument('--workers', type=int, default=4,
						metavar='N', help='dataloader threads')
	parser.add_argument('--base-size', type=int, default=512,
						help='base image size')
	parser.add_argument('--crop-size', type=int, default=512,
						help='crop image size')
	parser.add_argument('--sync-bn', type=bool, default=None,
						help='whether to use sync bn (default: auto)')
	parser.add_argument('--freeze-bn', type=bool, default=False,
						help='whether to freeze bn parameters (default: False)')
	parser.add_argument('--loss-type', type=str, default='ce',
						choices=['ce', 'focal'],
						help='loss func type (default: ce)')
	# training hyper params
	parser.add_argument('--epochs', type=int, default=None, metavar='N',
						help='number of epochs to train (default: auto)')
	parser.add_argument('--start_epoch', type=int, default=0,
						metavar='N', help='start epochs (default:0)')
	parser.add_argument('--batch-size', type=int, default=8,
						metavar='N', help='input batch size for \
								training (default: auto)')
	parser.add_argument('--test-batch-size', type=int, default=None,
						metavar='N', help='input batch size for \
								testing (default: auto)')
	parser.add_argument('--use-balanced-weights', action='store_true', default=True,
						help='whether to use balanced weights (default: False)')
	# optimizer params
	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (default: auto)')
	parser.add_argument('--lr-scheduler', type=str, default='poly',
						choices=['poly', 'step', 'cos'],
						help='lr scheduler mode: (default: poly)')
	parser.add_argument('--momentum', type=float, default=0.9,
						metavar='M', help='momentum (default: 0.9)')
	parser.add_argument('--weight-decay', type=float, default=5e-4,
						metavar='M', help='w-decay (default: 5e-4)')
	parser.add_argument('--nesterov', action='store_true', default=False,
						help='whether use nesterov (default: False)')
	# cuda, seed and logging
	parser.add_argument('--no-cuda', action='store_true', default=
						False, help='disables CUDA training')
	parser.add_argument('--gpu-ids', type=str, default='0,1',
						help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0,1)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	# checking point
	parser.add_argument('--resume', type=str, default=None,
						help='put the path to resuming file if needed')
	parser.add_argument('--checkname', type=str, default=None,
						help='set the checkpoint name')
	# finetuning pre-trained models
	parser.add_argument('--ft', type=bool, default=True,
						help='finetuning on a different dataset')
	# evaluation option
	parser.add_argument('--eval-interval', type=int, default=2,
						help='evaluuation interval (default: 1)')
	parser.add_argument('--no-val', type=bool, default=False,
						help='skip validation during training')
	parser.add_argument('--save-all', type=bool, default=True)

	parser.add_argument('--mode',type=str,help='options=train/val/test')

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

	if args.sync_bn is None:
		if args.cuda and len(args.gpu_ids) > 1:
			args.sync_bn = True
		else:
			args.sync_bn = False

	# default settings for epochs, batch_size and lr
	if args.epochs is None:
		epoches = {
			'coco': 30,
			'cityscapes': 200,
			'pascal': 50,
			'small_obstacle': 30
		}
		args.epochs = epoches[args.dataset.lower()]

	if args.batch_size is None:
		args.batch_size = 4 * len(args.gpu_ids)

	if args.test_batch_size is None:
		args.test_batch_size = args.batch_size

	if args.lr is None:
		lrs = {
			'coco': 0.1,
			'cityscapes': 0.01,
			'pascal': 0.007,
			'small_obstacle': 0.01
		}
		# args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
		args.lr = 0.01

	if args.checkname is None:
		args.checkname = 'deeplab-'+str(args.backbone)
	torch.manual_seed(args.seed)
	trainer = Trainer(args)
	print('Starting Epoch:', trainer.args.start_epoch)
	print('Total Epoches:', trainer.args.epochs)

	if args.mode=="train":
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
			trainer.training(epoch)
			if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
				trainer.validation(epoch)

	elif args.mode=="val" or args.mode=="test":

		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

			trainer.validation(epoch)

	trainer.writer.close()

if __name__ == "__main__":
	main()
