import os
import argparse
import torch
import torch.nn.functional as F
from model import rdbnn
from functional_networks import MLP_DNI_FCx3, CNN_DNI_CONVx2_FCx1
from dataset import *

# args
parser = argparse.ArgumentParser(description='DNI')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--conditioned', action="store_true")
parser.add_argument('--do_bn', action="store_true")
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--n_inner', type=int, default=1)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--beta', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=3e-5)
args = parser.parse_args()

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cuda')

# file name
model_name = '%s.beta%.4f_dni' % (args.dataset, args.beta)
if args.conditioned:
    model_name += '.conditioned'
args.model_name = model_name

# data
if args.dataset == 'mnist':
    data = mnist(args)
elif args.dataset == 'cifar10':
    data = cifar10(args)

train_loader = data.train_loader
test_loader = data.test_loader

# model
net_arch = MLP_DNI_FCx3 if args.dataset == 'mnist' else CNN_DNI_CONVx2_FCx1
net_args = dict(input_dim=data.in_channel, input_size=data.input_dims, device=device,
                n_hidden=args.n_hidden, n_classes=data.num_classes,
                conditioned_DNI=args.conditioned)
model = rdbnn(net_arch, net_args, F.nll_loss,
              do_bn=args.do_bn, lr=args.lr, n_inner=args.n_inner)

# main loop
best_perf = 0.
for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        loss, theta_loss, grad_loss = model.train_step(images, labels, beta=args.beta)

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Theta Loss: %4f, Grad Loss: %.4f'
                    % (epoch+1, args.num_epochs, i+1, data.num_train//args.batch_size,
                       loss, theta_loss, grad_loss))

    if (epoch+1) % 10 == 0:
        perf = model.test(test_loader, epoch+1, beta=args.beta)
    #    if perf[0] > best_perf:
    #        torch.save(model.net.state_dict(), model_name+'_model_best.pkl')

## Save the Model ans Stats
#pkl.dump(self.stats, open(self.model_name+'_stats.pkl', 'wb'))
#torch.save(self.net.state_dict(), self.model_name+'_model.pkl')
#if self.plot:
#    plot(self.stats, name=self.model_name)
