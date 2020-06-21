import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

## Load mnist dataset
download = False
current_dir = './data'
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dataset.MNIST(root=current_dir, train=True, transform=trans, download=download)
test_set = dataset.MNIST(root=current_dir, train=False, transform=trans)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

print '==>>> total trainning batch number: {}'.format(len(train_loader))
print '==>>> total testing batch number: {}'.format(len(test_loader))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,(5,5),padding=2)
        self.conv2d = nn.Conv2d(6,16,(5,5))
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2d(x)),(2,2))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(F.relu(self.fc2(x))))
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
net = LeNet()
print(net)
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
for epoch in xrange(10):
    ## Training
    for batch_idx, (x,target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x.cuda)