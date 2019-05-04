import torch.nn as nn
import torch.nn.Parameter as Parameter
from dni import *
import itertools
from functional_networks import mnist_mlp_dni
from nested_dict import nested_dict

# CNN Model (2 conv layer)
class cnn(nn.Module):
    def __init__(self, in_channel, conditioned_DNI, num_classes):
        super(cnn, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_classes)

        # DNI module
        self._layer1 = dni_Conv2d(16, (14, 14), num_classes, conditioned=conditioned_DNI)
        self._layer2 = dni_Conv2d(32, (7, 7), num_classes, conditioned=conditioned_DNI)
        self._fc = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)

        self.cnn = nn.Sequential(
                   self.layer1,
                   self.layer2,
                   self.fc)
        self.dni = nn.Sequential(
                   self._layer1,
                   self._layer2,
                   self._fc)
        self.optimizers = []
        self.forwards = []
        self.init_optimzers()
        self.init_forwards()

    def init_optimzers(self, learning_rate=0.001):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_fc)

    def forward_layer1(self, x, y=None):
        out = self.layer1(x)
        grad = self._layer1(out, y)
        return out, grad

    def forward_layer2(self, x, y=None):
        out = self.layer2(x)
        grad = self._layer2(out, y)
        return out, grad

    def forward_fc(self, x, y=None):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        grad = self._fc(out, y)
        return out, grad

    def forward(self, x, y=None):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer2_flat = layer2.view(layer2.size(0), -1)
        fc = self.fc(layer2_flat)
        if y is not None:
            grad_layer1 = self._layer1(layer1, y)
            grad_layer2 = self._layer2(layer2, y)
            grad_fc = self._fc(fc, y)
            return (layer1, layer2, fc), (grad_layer1, grad_layer2, grad_fc)
        else:
            return layer1, layer2, fc

# Neural Network Model (1 hidden layer)
class mlp(nn.Module):
    def __init__(self, conditioned_DNI, input_size, num_classes, lr=3e-5, hidden_size=256):
        super(mlp, self).__init__()

        # params
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.hidden_size = hidden_size

        # classify network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # dni network
        self._fc1 = dni_linear(hidden_size, num_classes, conditioned=conditioned_DNI)
        self._fc2 = dni_linear(num_classes, num_classes, conditioned=conditioned_DNI)
        # m weights
        params = itertools.chain(self.fc1.parameters(), self.fc2.parameters())
        self.m_mu = [torch.nn.Parameter(torch.zeros_like(w)) for w in params]
        self.m_rho = [torch.nn.Parameter(torch.log(torch.ones_like(w).exp()-1)) for w in params]

        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
        self.dni = nn.Sequential(self._fc1, self._fc2)
        self.optimizers = []
        self.forwards = []
        self.m_list = []
        self.init_optimzers()
        self.init_forwards()

    def init_optimzers(self):
        learning_rate = self.lr
        self.optimizers.append(torch.optim.Adam(self.fc1.parameters(), lr=learning_rate))
        self.optimizers.append(torch.optim.Adam(self.fc2.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.dni.parameters(), lr=learning_rate)
        self.m_optimizer = torch.optim.Adam(itertools.chain(self.m_mu, self.m_rho), lr=learning_rate)

    def init_forwards(self):
        self.forwards.append(self.forward_fc1)
        self.forwards.append(self.forward_fc2)

    def forward_fc1(self, x, y=None):
        x = x.view(-1, self.input_size)
        out = self.fc1(x)
        grad = self._fc1(out, y)
        return out, grad

    def forward_fc2(self, x, y=None):
        x = self.relu(x)
        out = self.fc2(x)
        grad = self._fc2(out, y)
        return out, grad

    def forward(self, x, y=None):
        x = x.view(-1, self.input_size)
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)

        if y is not None:
            grad_fc1 = self._fc1(fc1, y)
            grad_fc2 = self._fc2(fc2, y)
            return (fc1, fc2), (grad_fc1, grad_fc2)
        else:
            return fc1, fc2


class rdbnn(nn.Module):
    def __init__(self, input_dim=1, input_size=28*28, device='cpu', do_bn=False,
              n_hidden=400, n_classes=10, lr=3e-5, conditioned_DNI=True, n_inner=3):
        super(rdbnn, self).__init__()

        # params
        self.input_dim = input_dim
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_inner = n_inner

        # functional network
        self.net = mnist_mlp_dni(
                input_dim=input_dim, input_size=input_size, device=device, do_bn=do_bn,
                n_hidden=n_hidden, n_classes=n_classes, dni_class=dni_linear, conditioned_DNI)

        # m_psi (Gaussian) and intermediate theta
        # TODO: m to be MLP or ConvNet
        self.m_mu, self.m_rho, self.inter_theta = {}, {}, {}
        for l, layer in self.net.params.items():
            self.m_mu[l] = {}
            self.m_rho[l] = {}
            self.inter_theta[l] = {}
            for k, w in layer.items():
                self.m_mu[l][k] = Parameter(torch.zeros_like(w, device=device)).requires_grad_()
                self.m_rho[l][k] = Parameter(torch.log(torch.ones_like(w, device=device).exp()-1)).requires_grad_()
                self.register_parameter(l+'_'+k, self.m_mu[l][k])
                self.register_parameter(l+'_'+k, self.m_rho[l][k])

        # optimizers
        self.theta_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.grad_optimizer = torch.optim.Adam(self.net.dni_parameters(), lr=self.lr)
        #self.m_optimizer = torch.optim.Adam(itertools.chain(self.m_mu.values(), self.m_rho.values()), lr=self.lr)
        self.m_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def refine_theta(self, key, input, activation, label_onehot=None):
        '''
        The graph starts with theta = net.params[key]
        (note that '=' will also copy .grad, so zero_grad() should be called).
        theta is refined by GD with approximate gradients from DNI,
        where input is detached from previous activation.
        Finally, theta will be used in self.net.forward(theta, input)
        '''
        self.theta_optimizer.zero_grad()
        theta = self.net.init_theta(key) # {weight, bias} for a fc

        for t in range(self.n_inner):
            out, grad = self.net.f_fc(input, theta, key, activation, label=label_onehot, training=True)
            out.backward(grad, create_graph=True, retain_graph=True) # TODO: try grad.detach() or freeze dni?

            # loss m
            loss_m = neg_log_m(theta)
            loss_m.backward(create_graph=True, retain_graph=True)

            # GD
            for k, w in theta.items():
                assert(w.grad in not None)
                theta[k] = w - self.lr * w.grad # TODO: try diff lr

        self.inter_theta[key] = theta

        return out.detach()

    def forward(self, x, y=None, training=True):
        theta = {k:v.detach().requires_grad_() for k,v in theta.items()}

        for t in range(self.n_inner):
            logit, grads = self.f(x, theta, self.bn_stats, training, dni=self.dni, label=y)
            for k, grad in grads.items():
                theta[k] =

        return logit, grads

    def neg_log_m(self, theta, key):
        c = - float(0.5 * math.log(2 * math.pi))
        std_w = (1 + self.m_rho[key]['weight'].exp()).log() # TODO: need to make std larger?
        logvar_w = std_w.pow(2).log()
        logpdf_w = c - 0.5 * logvar_w -
            (theta['weight'] - self.m_mu[key]['weight']).pow(2) / (2 * std_w.pow(2))
        std_b = (1 + self.m_rho[key]['bias'].exp()).log()
        logvar_b = std_b.pow(2).log()
        logpdf_b = c - 0.5 * logvar_b -
            (theta['bias'] - self.m_mu[key]['bias']).pow(2) / (2 * std_b.pow(2))


