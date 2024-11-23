import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input


class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a* b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)


class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, image):
		return (image - self.mean) / self.std


def style_cnn(cnn, device, normalization_mean, normalization_std, style_image, content_image):

	style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
	content_layers = ['conv_4']
	cnn = copy.deepcopy(cnn)
	style_losses = []
	content_losses = []

	normalization = Normalization(normalization_mean, normalization_std).to(device)
	model = nn.Sequential(normalization)
	i = 0

	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer = nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn_{}'.format(i)

	
		model.add_module(name, layer)

	
		if name in style_layers:
			target_feature = model(style_image).detach()
			style_loss = StyleLoss(target_feature)
			model.add_module('style_loss_{}'.format(i), style_loss)
			style_losses.append(style_loss)

	
		if name in content_layers:
			target = model(content_image).detach()
			content_loss = ContentLoss(target)
			model.add_module('content_loss_{}'.format(i), content_loss)
			content_losses.append(content_loss)


	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], StyleLoss) or isinstance(model[i], ContentLoss):
			break

	model = model[:(i + 1)]

	return model, style_losses, content_losses