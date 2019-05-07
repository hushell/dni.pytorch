import torch.nn as nn
import torch.autograd as autograd
from torch.nn import Parameter
import torch.nn.functional as F
import itertools
import math


class rdbnn(nn.Module):
    def __init__(self, net_arch, net_args, task_loss,
                 do_bn=False, lr=3e-5, n_inner=2):
        super(rdbnn, self).__init__()

        # params
        self.task_loss = task_loss
        self.lr = lr
        self.n_inner = n_inner
        self.n_classes = net_args['n_classes']
        self.device = net_args['device']
        self.cond_dni = net_args['conditioned_DNI']

        # functional network
        self.net = net_arch(**net_args)

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
                self.register_parameter(l+'_'+k+'_mu', self.m_mu[l][k])
                self.register_parameter(l+'_'+k+'_rho', self.m_rho[l][k])

        # optimizers
        self.theta_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.grad_optimizer = torch.optim.Adam(self.net.dni_parameters(), lr=self.lr)
        self.m_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def refine_theta(self, key, input, y_onehot=None, training=True, beta=1.0):
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
            out, grad, _ = self.net.layer(key, theta, input,
                                          y_onehot=y_onehot, do_grad=True, training=training)

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
        input = self.net.preprocess(x)

        # obtain refined theta and store in self.inter_theta
        for key in self.net.params.keys():
            input = self.refine_theta(key, input, y_onehot, training, beta)

        logits = self.net.forward(self.inter_theta, x, y_onehot,
                                  do_grad=False, training=True)
        return logits

    def neg_log_m(self, theta, key):
        c = -float(0.5 * math.log(2 * math.pi))
        std_w = (1 + self.m_rho[key]['weight'].exp()).log() # TODO: make std larger avoid underflow?
        logvar_w = std_w.pow(2).log()
        logpdf_w = c - 0.5 * logvar_w - \
            (theta['weight'] - self.m_mu[key]['weight']).pow(2) / (2 * std_w.pow(2))
        std_b = (1 + self.m_rho[key]['bias'].exp()).log()
        logvar_b = std_b.pow(2).log()
        logpdf_b = c - 0.5 * logvar_b - \
            (theta['bias'] - self.m_mu[key]['bias']).pow(2) / (2 * std_b.pow(2))
        return -0.5 * (logpdf_w.mean() + logpdf_b.mean())

    def train_step(self, x, y, beta=1.0):
        x, y = x.to(self.device), y.to(self.device)

        # y_onehot for dni
        if self.cond_dni:
            y_onehot = torch.zeros([y.size(0), self.n_classes]).to(self.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
        else:
            y_onehot = None

        # update theta (or init_net) and m_psi
        self.theta_optimizer.zero_grad()
        self.m_optimizer.zero_grad()

        logits = self.forward(x, y, y_onehot, training=True, beta=beta)

        nll = self.task_loss(logits, y)
        kl = sum([self.neg_log_m(theta, key) for key, theta in self.inter_theta.items()])
        loss = nll + beta * kl
        loss.backward()

        self.theta_optimizer.step()
        self.m_optimizer.step()

        # update dni
        self.theta_optimizer.zero_grad() # clean theta.grad
        self.grad_optimizer.zero_grad()

        theta_loss, grad_loss = self.update_dni_module(x, y, y_onehot)

        grad_loss.backward()
        self.grad_optimizer.step()

        return loss.item(), theta_loss.item(), grad_loss.item()

    def test(self, test_loader, epoch, beta=1.0):
        correct = [0, 0]
        total = 0

        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)

            # y_onehot for dni
            if self.cond_dni:
                y_onehot = torch.zeros([y.size(0), self.n_classes]).to(self.device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1)
            else:
                y_onehot = None

            # with refined theta
            logits_refine = self.forward(x, y, y_onehot, training=False, beta=beta)
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
        print('==> Test at Epoch %d: [with refinement: %.4f] - [normal: %.4f]' % (epoch, perf[0], perf[1]))
        return perf

    def update_dni_module(self, x, y, y_onehot):
        self.net.dni_seq.train()

        # forward with self.net.params
        logits, grads, fcs = self.net.forward(self.net.params, x, y_onehot,
                                              do_grad=True, training=False)

        # register hooks
        real_grads = {}
        handles = {}

        def save_grad(key):
            def hook(grad):
                real_grads[key] = grad
            return hook

        for key, fc in fcs.items():
            handles[key] = fc.register_hook( save_grad(key) )

        # compute real grads
        loss = self.task_loss(logits, y)
        loss.backward(retain_graph=True) # need to backward again

        # remove hooks
        for v in handles.values():
            v.remove()

        # dni loss & step
        grad_loss = sum([F.mse_loss(grads[key], real_grads[key].detach())
                         for key in fcs.keys()])

        self.net.dni_seq.eval()
        return loss, grad_loss

