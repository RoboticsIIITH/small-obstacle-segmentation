import torch
import sys
sys.path.append('/home/ash/Small-Obs-Project/Small_Obstacle_Segmentation')
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
	def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
				 sync_bn=True, freeze_bn=False):
		super(DeepLab, self).__init__()
		if backbone == 'drn':
			output_stride = 8

		if sync_bn == True:
			BatchNorm = SynchronizedBatchNorm2d
		else:
			BatchNorm = nn.BatchNorm2d

		self.backbone = build_backbone(backbone, output_stride, BatchNorm)
		self.aspp = build_aspp(backbone, output_stride, BatchNorm)
		self.decoder = build_decoder(num_classes, backbone, BatchNorm)

		if freeze_bn:
			self.freeze_bn()

	def forward(self, input,reg_prop):
		input = torch.cat((input,reg_prop),dim=1)
		x, low_level_feat = self.backbone(input)
		x = self.aspp(x)
		x = self.decoder(x, low_level_feat)
		x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

		return x

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, SynchronizedBatchNorm2d):
				m.eval()
			elif isinstance(m, nn.BatchNorm2d):
				m.eval()

	def get_1x_lr_params(self):
		modules = [self.backbone]
		for i in range(len(modules)):
			for m in modules[i].named_modules():
				if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
						or isinstance(m[1], nn.BatchNorm2d):
					for p in m[1].parameters():
						if p.requires_grad:
							yield p

	def get_10x_lr_params(self):
		modules = [self.aspp, self.decoder]
		for i in range(len(modules)):
			for m in modules[i].named_modules():
				if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
						or isinstance(m[1], nn.BatchNorm2d):
					for p in m[1].parameters():
						if p.requires_grad:
							yield p


if __name__ == "__main__":

	# model = DeepLab(backbone='drn', output_stride=16, num_classes=3)
	# model.train()
	# image = torch.rand(2, 3, 512, 512)
	# depth = torch.rand(2, 1, 512, 512)
	# depth_mask = depth != 0
	# depth_mask = depth_mask.float()
	checkpoint = torch.load('/home/ash/Small-Obs-Project/deeplab_checkpoints/checkpoints/deeplab_4_channel_inp.pth',map_location='cpu')
	# checkpoint_2 = torch.load('/home/ash/Small-Obs-Project/nconv/workspace/exp_guided_enc_dec/unguided_network_pretrained/CNN_ep0005.pth.tar')
	# depth_layers = checkpoint_2['net']
	# new_depth_layers = depth_layers.copy()
	# for key,value in iter(depth_layers.items()):
	# 	new_key = 'depth_backbone.' + str(key)
	# 	new_depth_layers[new_key] = value
	# 	del new_depth_layers[key]
	#
	# print(new_depth_layers.keys())
	# checkpoint['state_dict'].update(new_depth_layers)
	# torch.save(checkpoint,'/home/ash/Small-Obs-Project/deeplab_checkpoints/deeplab_5_channel.pth')
	# print(summary(model,[(3,512,512),(1,512,512),(1,512,512)],batch_size=2))
	# model.load_state_dict(checkpoint['state_dict'])
	# output = model(image, depth, depth_mask)
	# for key in checkpoint['state_dict'].keys():
	# 	print(key, checkpoint['state_dict'][key].shape)
	# for i,layer in enumerate(checkpoint['state_dict'].keys()):
	# 	print(i," ",layer," ",checkpoint["state_dict"][layer].shape)
	first_layer_weight = checkpoint['state_dict']['backbone.layer0.0.weight']
	print(first_layer_weight.shape)
	# weight_shape = [16,1,7,7]
	# new_first_layer = nn.init.kaiming_normal_(torch.empty(weight_shape))
	# new_first_layer = torch.cat((first_layer_weight,new_first_layer),dim=1)
	# checkpoint['state_dict']['backbone.layer0.0.weight'] = new_first_layer
	# torch.save(checkpoint,'/home/ash/Small-Obs-Project/deeplab_checkpoints/checkpoints/deeplab_4_channel_inp.pth')