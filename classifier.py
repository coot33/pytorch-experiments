import torch
import torchvision
import torchvision.transforms as transforms



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


import torch.optim as optim

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train = True

    if train:
        for epoch in range(2):
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)
                #inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                output = net(inputs)

                loss = criterion(output, labels)


                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))

                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), 'cifar-10-model')

    net.load_state_dict(torch.load('cifar-10-model'))

    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))


    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, targets = data

        images = Variable(inputs)
        labels = Variable(labels)
        #images = Variable(inputs)

        output = net(images)

        _, predicted = torch.max(output.data, 1)
        c = predicted.eq(labels.data).squeeze()
        for i in range(4):
            label = targets[i]
            class_correct[label] += c[i]
            class_total[label] += 1
            total += 1
            correct += c[i]

    print('Total test accuracy : %d %%' %
          (100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))






