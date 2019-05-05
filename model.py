import torch.nn as nn
import torch.autograd as autograd
import torch.nn.Parameter as Parameter
from dni import *
import itertools
from functional_networks import mnist_mlp_dni


class rdbnn(nn.Module):
    def __init__(self, task_loss, input_dim=1, input_size=28*28, device='cpu', do_bn=False,
              n_hidden=400, n_classes=10, lr=3e-5, conditioned_DNI=True, n_inner=2):
        super(rdbnn, self).__init__()

        # params
        self.task_loss = task_loss
        self.input_dim = input_dim
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_inner = n_inner
        self.device = device
        self.cond_dni = conditioned_DNI

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

    def refine_theta(self, key, input, y_onehot=None, beta=1.0):
        '''
        The graph starts with theta = net.params[key]
        (note that '=' will also copy .grad, so zero_grad() should be called).
        theta is refined by GD with approximate gradients from DNI,
        where input is detached from previous activation.
        Finally, theta will be used in self.net.forward(theta, input)
        '''
        theta = self.net.init_theta(key) # {weight, bias} for a fc

        for t in range(self.n_inner):
            # fc_i(theta)
            out, grad, _ = self.net.f_fc(key, theta, input,
                                         y_onehot=y_onehot, do_grad=True, training=True)

            # -log m_psi(theta)
            loss_m = beta * self.neg_log_m(theta, key)

            # compute grads
            grad_theta = autograd.grad(outputs=[out, loss_m], inputs=theta.values(),
                                        grad_outputs=[grad, torch.ones_like(loss_m)],
                                        create_graph=True, retain_graph=True)
            # GD
            for i, (k, w) in enumerate(theta.items()):
                theta[k] = w - self.lr * grad_theta[i] # TODO: try diff lr or lr param

        # store refined theta
        self.inter_theta[key] = theta

        return out.detach() # make `out' a leaf

    def forward(self, x, y, y_onehot=None, training=True, beta=1.0):
        '''
        forward with refined theta
        '''
        # obtain refined theta and store in self.inter_theta
        input = x
        for key in self.net.params.keys():
            input = self.refine_theta(key, input, y_onehot, beta)

        logits = self.net.forward(self.inter_theta, x, y_onehot,
                                  do_grad=False, training=True)
        return logits

    def neg_log_m(self, theta, key):
        c = -float(0.5 * math.log(2 * math.pi))
        std_w = (1 + self.m_rho[key]['weight'].exp()).log() # TODO: need to make std larger?
        logvar_w = std_w.pow(2).log()
        logpdf_w = c - 0.5 * logvar_w -
            (theta['weight'] - self.m_mu[key]['weight']).pow(2) / (2 * std_w.pow(2))
        std_b = (1 + self.m_rho[key]['bias'].exp()).log()
        logvar_b = std_b.pow(2).log()
        logpdf_b = c - 0.5 * logvar_b -
            (theta['bias'] - self.m_mu[key]['bias']).pow(2) / (2 * std_b.pow(2))
        return -(logpdf_w + logpdf_b)

    def train_step(self, x, y, beta=1.0):
        x, y = x.to(self.device), y.to(self.device)

        # y_onehot for dni
        if self.cond_dni:
            y_onehot = torch.zeros([y.size(0), self.n_classes]).to(self.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
        else:
            y_onehot = None

        # update theta (or init_net) and m_psi
        theta_optimizer.zero_grad()
        m_optimizer.zero_grad()

        logits = self.forward(x, y, y_onehot, training=True, beta)

        nll = self.task_loss(logits, y)
        kl = sum([self.neg_log_m(theta, key) for key, theta in self.inter_theta.items()])
        loss = nll + beta * kl
        loss.backward()

        theta_optimizer.step()
        m_optimizer.step()

        # update dni
        theta_optimizer.zero_grad() # clean theta.grad
        grad_optimizer.zero_grad()

        theta_loss, grad_loss = self.net.update_dni_module(x, y, y_onehot,
                self.task_loss, self.theta_optimizer, self.grad_optimizer)

        grad_loss.backward()
        grad_optimizer.step()

        return loss.item(), theta_loss.item(), grad_loss.item()

    def test(self, test_loader, epoch, beta=1.0):
        correct = [0, 0]
        total = 0

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)

            # y_onehot for dni
            if self.cond_dni:
                y_onehot = torch.zeros([y.size(0), self.n_classes]).to(self.device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1)
            else:
                y_onehot = None

            # with refined theta
            logits_refine = self.forward(x, y, y_onehot, False, beta)
            _, predicted = torch.max(logits_refine, 1)
            total += y.size(0)
            correct[0] += (predicted == y).sum().item()

            # TODO: check the diff between logits and logits_refine
            # with refined theta
            with torch.no_grad():
                logits = self.net.forward(self.net.params, x, training=False)
                _, predicted = torch.max(logits, 1)
                correct[1] += (predicted == y).sum().item()

        perf = [100 * correct[0] / total, 100 * correct[1] / total]
        print('Epoch %d: [with refinement: %.4f] - [normal: %.4f]' % (epoch, perf[0], perf[1]))
        return perf

