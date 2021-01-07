import argparse
import pickle
import vgg
import torch
import torch.optim as optim
from torch.utils import data
from HDF5Dataset import HDF5Dataset
import sys
# define the command line parser.
parser = argparse.ArgumentParser(description="""Script to train the GalaxyZoo VGG network on the
                                GalaxyZoo dataset.""")
parser.add_argument('--batch-size', '-b', type=int, default=16, help='batch size for training.')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs for training.')
parser.add_argument('--dataset-path', type=str, default='./data', help='path to the galaxy zoo dataset.')
parser.add_argument('--optimizer', type=str.lower, default='sgd', help='optimizer for training.')
parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate for training.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for training.')
parser.add_argument('--network', type=str, default='vgg11', help='CNN network to train.')
parser.add_argument('--batch-norm', default=False, action='store_true')
parser.add_argument('--data-path', type=str, default='/home/idies/workspace/Temporary/adi/scratch/galaxyzoo-cnn-aas2021/data/', help='Path to the galaxyzoo dataset.')
args = parser.parse_args()

# check args values.
if args.optimizer != 'sgd' and args.optimizer != 'adam':
    raise ValueError("--optimizer must be 'sgd' or 'adam'. Got '{}' instead.".format(args.optimizer))
if args.batch_norm:
    args.network += '_bn'
vgg_network = getattr(vgg, args.network)()

if args.optimizer == 'sgd':
    optimizer = optim.SGD(vgg_network.parameters(), lr=args.learning_rate, momentum=args.momentum)
print('{}\n'.format(vars(args)))
print('{}\n'.format(vgg_network))
print('{}\n'.format(optimizer))

train_data = HDF5Dataset(args.data_path, min_pixel_dims=0, max_pixel_dims=sys.maxsize)
data_loader = data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg_network.to(device)
for epoch in range(100):
    running_loss = 0.
    nitems = 0
    total_loss = 0.
    train_acc = 0.
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data[0].float().to(device), data[1].float().to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_logsftmx = F.log_softmax(outputs, dim=-1)
        pred = torch.argmax(outputs_logsftmx, dim=1)
        ground_truth = torch.argmax(labels, dim=1)
        train_acc += torch.sum(pred == ground_truth).item()
        loss = torch.mean(torch.sum(- labels * outputs_logsftmx, 1))
        loss.backward()
        optimizer.step()
        nitems += 1
        running_loss += loss.item()
        
    print('[%d] loss: %.5f, training accuracy: %.2%%' %(epoch+1, running_loss, 100*train_acc))
    running_loss = 0.
#for epoch in range(args.epochs):
#    for i, data in enumerate(data_loader, 0):
