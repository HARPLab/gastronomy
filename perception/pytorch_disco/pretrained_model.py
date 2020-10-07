import torch
import torch.nn as nn
# state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
import torchvision.models as models

import ipdb
st = ipdb.set_trace

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		# image_modules = list(models.resnet18().children())[:-6] #all layer expect last layer
		# image_modules = list(models.alexnet().children())[:-6] #all layer expect last layer
		# image_modules = list(list(models.alexnet().children())[:1][0].children())[:1]
		self.modelFirst = nn.Sequential(*list(list(models.resnet18().children())[:4]))
		self.modelSecond =list(list(models.resnet18().children())[4].children())[0]
		self.modelThird = list(list(models.resnet18().children())[4].children())[1]
		
	def forward(self, image):
		val = self.modelFirst(image)
		val1 = self.modelSecond(val)
		val2 = self.modelThird(val1)
		value = torch.cat([val,val1,val2],dim=1)
		return value

if __name__ == "__main__":
	model = MyModel()
	model.cuda()
	img = torch.randn(1,3,256,256).cuda() 
	print(model(img).shape)
	st()
	print("check")
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)