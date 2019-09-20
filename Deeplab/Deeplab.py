from torchvision.models.segmentation import deeplabv3_resnet101
from torch.utils.data import DataLoader
from torch.nn import *
import torch
import Image.imgstream as imgstream
from Dataset import *

class Deeplab:

	def __init__(self):
		self.dataset = ImageDataset(switchdim=True, tofloat=True, normalize=True)
		self.model = self.define_model()
		self.lossfn = nn.MSELoss(reduction='mean')
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

	def define_model(self):
		model = deeplabv3_resnet101(pretrained=True)
		model.classifier[-1] = Conv2d(256, 30, 1)
		return model

	def train(self, dataset=None, batch_size=16, epochs=100):
		if dataset is not None:
			self.dataset = dataset
		self.model.train()
		train_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)
		for n in range(epochs):
			for x_batch, y_batch in train_loader:
				yhat_batch = self.model(x_batch)
				loss = self.lossfn(y_batch, yhat_batch)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

	def eval(self, x, argmax=False, transform=True):
		self.model.eval()
		batch = len(x.shape) > 3
		x = torch.tensor(x)
		if not batch:
			x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
		if transform:
			x = self.dataset.transform(x)
		y = self.model(x)['out']
		if argmax:
			y = y.argmax(1)
		if not batch:
			y = y[0,:,:,:]
		return y

	def __getitem__(self,x):
		return self.eval(x)


if __name__ == '__main__':
	segmenter = Deeplab()
	stream = imgstream.Stream(mode='webcam',src="")
	for img in stream:
		mask = segmenter.eval(img)
		imgstream.Stream.show(img, name="mask", pause=False)