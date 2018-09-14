import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()

class Dataset(Dataset):
    def __init__(self, pt_file_path, transform = None):
        self.data, self.labels = torch.load(pt_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_label = self.labels[index]
        if self.transform:
            sample_data = self.transform(sample_data.unsqueeze(2).numpy())

        return sample_data, sample_label

num_epochs = 4

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor()])

# Dataloader 파라매터
params = {'batch_size': 4,
          'shuffle': False}
training_set, validation_set = Dataset('./data/train.pt', transform), Dataset('./data/valid.pt', transform)
train_loader = torch.utils.data.DataLoader(training_set, **params)
valid_loader = torch.utils.data.DataLoader(validation_set, **params)

class FASHION_MNIST_Net(nn.Module): #https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5
    def __init__(self):
        super(FASHION_MNIST_Net, self).__init__()
        # an affine operation: y = Wx + b
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        # Sequential. 예를 들어, 회선 층은 다음과 같이 정의됩니다 nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride) . 두 개의 컨볼 루션 계층과 활성화 후에 네트워크를 10 개의 클래스로 출력하는 완전 연결 계층으로 종료합니다.
    def forward(self, x): # forward함수는 일련의 입력을 위해 신경망에서 호출되며 정의 된 다른 계층을 통해 해당 입력을 전달합니다.
        out = self.layer1(x) # 이 경우 x첫 번째 레이어를 통과하고 두 번째 레이어를 통해 출력을 전달한 다음 마지막으로 완전히 연결된 레이어를 통과시켜 출력합니다
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

fashion_mnist_net = FASHION_MNIST_Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fashion_mnist_net.parameters(), lr=0.0001) # 0.0005 -> local minima? or globa

def train_network(net,optimizer,train_loader):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader): # 한 Epoch 만큼 돕니다. 매 iteration 마다 정해진 Batch size 만큼 데이터를 뱉습니다. 
            # get the inputs
            inputs, labels = data # DataLoader iterator의 반환 값은 input_data 와 labels의 튜플 형식입니다. 
            inputs = Variable(inputs)#.cuda() # Pytorch에서 nn.Module 에 넣어 Backprop을 계산 하기 위해서는 Variable로 감싸야 합니다.
            labels = Variable(labels)#.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()    #  현재 기존의 backprop을 계산하기 위해서 저장했던 activation buffer 를 비웁니다. Q) 이걸 안 한다면? -> 정확도가 점차 내려감? overfitting? buffer가 안비워지니 계속 중첩해서 쌓임

            # forward + backward + optimize
            outputs = net(inputs) # input 을 넣은 위 network 로 부터 output 을 얻어냅니다. 
            loss = criterion(outputs, labels) # loss fucntion에 주어진 target과 output 의 score를 계산하여 반환합니다. 
            loss.backward(retain_graph=True) # * Scalar Loss value를 Backward() 해주게 되면 주어진 loss값을 바탕으로 backpropagation이 진행됩니다. 
            optimizer.step() # 계산된 Backprop 을 바탕으로 optimizer가 gradient descenting 을 수행합니다. 


        # 검증 데이터 정확도 측정
        correct = 0
        total = 0

        for images, labels in valid_loader:
            images = Variable(images)
            labels = Variable(labels)

            # logit과 output을 얻기 위해 model의 Forward pass에 입력
            outputs = net(images)

            # 예측한 클래스 얻기
            _, predicted = torch.max(outputs.data, 1)

            # 미니배치의 검증 데이터 수
            total += labels.size(0)

            # 미니배치 중 맞은 갯수
            correct += (predicted == labels.data).sum()

        # 전체 검증 데이터 정확도
        accuracy = 100 * correct / total

        # loss 출력
        print('Epoch: {}. Training Loss: {}. Validation Accuracy: {}'.format(epoch, loss.data[0], accuracy))


    print('Finished Training')

def test(model,valid_loader):
    model.eval() # Eval Mode 왜 해야 할까요?  --> nn.Dropout BatchNorm 등의 Regularization 들이 test 모드로 들어가게 되기 때문입니다. 
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data), Variable(target)  # 기존의 train function의 data 처리부분과 같습니다. 
        output = model(data) 
        pred = output.max(1, keepdim=True)[1] # get the index of the max 
        correct += pred.eq(target.view_as(pred)).sum().data[0] # 정답 데이터의 갯수를 반환합니다. 

    test_loss /= len(valid_loader.dataset)
    print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    
train_network(fashion_mnist_net,optimizer,train_loader) # 4 Epoch 정도 학습을 진행해봅니다. 

test(fashion_mnist_net,valid_loader) # Test 정확도를 출력해 봅니다. 
