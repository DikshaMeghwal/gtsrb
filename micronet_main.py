from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
from torch.autograd import Variable
from logger import Logger

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--suffix', type=str, default='', metavar='D',
                    help='suffix for the filename of models and output files')

args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms, data_transforms1, data_transforms2, data_transforms3, data_transforms4, data_transforms5 # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

train_loader1 = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms1),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

train_loader2 = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms2),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

train_loader3 = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms3),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

train_loader4 = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms4),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

train_loader5 = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms5),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model import Net
from micronet_model import Net, lr, momentum, seed, l2_norm, batch_size, decay, step, epochs, log_interval 
model = Net()
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=lr , momentum=momentum, weight_decay=l2_norm, nesterov=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

# using tensorboard for visualization
#log_file_name = "bs:" + str(args.batch_size) + "_epochs:" + str(args.epochs) + "_lr:" + str(args.lr) + "_mom:" + str(args.momentum) + "_" + args.suffix
logger = Logger('./logs/' + 'micronet_' + args.suffix)
dtype = torch.cuda.FloatTensor
def train(epoch, train_loader):
    model.train()
    # print(model.parameters)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data.type(dtype))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), (epoch * args.batch_size) + batch_idx)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch + batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        output = model(data.type(dtype))
        validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    logger.scalar_summary("validation", validation_loss, epoch)
    logger.scalar_summary("accuracy", 100. * correct / len(val_loader.dataset), epoch)
    # log_value("validation loss", validation_loss, epoch)
    # log_value("accuracy", 100. * correct / len(val_loader.dataset), epoch)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss

step = 50
temp = 999

for epoch in range(1, args.epochs + 1):
    train(epoch, train_loader)
    #train(epoch, train_loader1)
    #train(epoch, train_loader2)
    #train(epoch, train_loader3)
    #train(epoch, train_loader4)
    #train(epoch, train_loader5)

    val = validation(epoch)
    if(epoch % step):
        scheduler.step()
    if(val < temp):
        model_file = 'micronet_model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')

