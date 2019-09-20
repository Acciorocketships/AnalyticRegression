import torch
from AtomicFunction import AtomicFunction



class Identification(torch.nn.Module):

	def __init__(self):
		super(Identification, self).__init__()
		self.sum_layer = AtomicFunction(self.sumfunc, 2, 2)
		self.cos_layer = AtomicFunction(self.cosfunc, 2, 1)
		self.mul_layer = AtomicFunction(self.mulfunc, 2, 2)
		self.dropout = torch.nn.Dropout(p=0.3)
		self.combination_layer= torch.nn.Linear(3,1)

	def forward(self,x):
		x1_1 = self.sum_layer(x)
		x1_2 = self.cos_layer(x)
		x1_3 = self.mul_layer(x)
		x2 = torch.cat((x1_1, x1_2, x1_3), dim=1)
		x2_dropout = self.dropout(x2)
		x3 = self.combination_layer(x2_dropout)
		return x3

	def sumfunc(self,x):
		return torch.sum(x, dim=1)

	def cosfunc(self,x):
		return torch.cos(x)

	def mulfunc(self,x):
		return torch.prod(x,dim=1)




# test function which we want to fit
def func(x):
	return 0.5*torch.cos(x[:,0]) + 0.3*x[:,1]


def add_noise(x):
	return x + torch.randn(x.shape) / 5


def train(model):
	batch_size = 4
	criterion = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

	iteration = 0
	ax = None
	loss_ema_long = 0
	loss_ema_short = 0
	while True:

		# Training mode
		model.train()

		# Evaluate with training data
		x = torch.rand(batch_size, 2) * 8 - 4
		y = func(x)
		y_pred = model(x)

		# Compute Loss
		loss = criterion(y, y_pred)
		loss_ema_long = 0.99 * loss_ema_long + 0.01 * loss
		loss_ema_short = 0.9 * loss_ema_short + 0.1 * loss
		print(loss)

		# Stopping condition
		if loss < 1E-5:
			break

		# Train model
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Helps escape local minima
		if iteration % 50000 == 0:
			if loss_ema_short >= 0.99 * loss_ema_long:
				with torch.no_grad():
					for param in model.parameters():
						param.add_(torch.randn(param.size()) * 0.1)

		# Display				
		if iteration % 1000 == 0:
			ax = eval(model, ax)

		iteration += 1

	print('done training')


def eval(model, ax=None, pause=False):
	model.eval()
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		plt.ion()
		plt.show()

	x1, x2 = torch.meshgrid([torch.arange(-4,4,step=0.25), torch.arange(-4,4,step=0.25)])
	x = torch.cat((x1.reshape(-1,1), x2.reshape(-1,1)), dim=1)
	y = func(x)
	y_2d = y.reshape(x1.shape)

	y_pred = model(x).detach().numpy()
	y_pred_2d = y_pred.reshape(x1.shape)

	plt.cla()
	ax.plot_surface(x1,x2,y_2d)
	ax.scatter(x1,x2,y_pred_2d, color='r')
	if pause:
		plt.ioff()
		plt.draw()
	else:
		plt.draw()
		plt.pause(0.001)
	return ax


if __name__ == '__main__':
	model = Identification() # test single layer to start
	train(model)
	eval(model, pause=True)

