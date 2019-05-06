import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, batch_norm
import itertools


class MNIST_MLP_DNI:
    def __init__(self, input_dim=1, input_size=28*28, device='cpu', do_bn=False,
                 n_hidden=400, n_classes=10, dni_class=None, conditioned_DNI=True):
        self.input_dim = input_dim
        self.input_size = input_size
        self.device = device
        self.do_bn = do_bn
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.dni_class = dni_class
        self.conditioned_DNI = conditioned_DNI

        def gen_params():
            params = {
                'fc1': linear_params(input_dim*input_size, n_hidden, device),
                'fc2': linear_params(n_hidden, n_hidden, device),
                'fc3': linear_params(n_hidden, n_classes, device)}
            activations = {'fc1': F.relu, 'fc2': F.relu, 'fc3': None}
            return params, activations

        def gen_bn_params():
            params = {'fc1': bnparams(n_hidden, device),
                      'fc2': bnparams(n_hidden, device)}
            return params

        def gen_bn_stats():
            stats = {'fc1': bnstats(n_hidden, device),
                     'fc2': bnstats(n_hidden, device)}
            return stats

        def gen_dni():
            dni = {
                'fc1': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI).to(device),
                'fc2': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI).to(device),
                'fc3': dni_class(n_classes, n_classes, conditioned=conditioned_DNI).to(device)}
            return dni

        # All params
        self.params, self.activations = gen_params()
        if do_bn:
            self.bn_params = gen_bn_params()
            self.bn_stats = gen_bn_stats()
        self.dni = gen_dni()

    def parameters(self):
        for layer in self.params.values():
            for w in layer.values():
                yield w
        if self.do_bn:
            for layer in self.bn_params.values():
                for w in layer.values():
                    yield w

    def dni_parameters(self):
        return itertools.chain(self.dni['fc1'].parameters(),
                               self.dni['fc2'].parameters(),
                               self.dni['fc3'].parameters())

    def init_theta(self, key):
        #return {k:v.detach().requires_grad_() for k,v in self.params[key].items()}
        return {k:v.clone() for k,v in self.params[key].items()}

    def f_fc(self, key, theta, input, y_onehot=None, do_grad=False, training=True):
        '''
        TODO: y_onehot=None and conditioned_DNI = False
        '''
        # linear
        fc = F.linear(input, theta['weight'], theta['bias'])

        # dni
        if do_grad:
            grad = self.dni[key](fc, y_onehot) # d_loss/d_fc
        else:
            grad = None

        # activation
        output = fc if self.activations[key] is None else self.activations[key](fc)

        # batchnorm
        if self.do_bn and self.activations[key] is not None:
            output = batch_norm(output, self.bn_params, self.bn_stats, key, training)

        return output, grad, fc

    def forward(self, params, x, y_onehot=None, do_grad=False, training=True):
        '''
        forward without theta refinement
        '''
        grads, fcs = {}, {}

        input = x.view(-1, self.input_dim*self.input_size)

        for key in self.params.keys():
            input, grad, fc = self.f_fc(key, params[key], input, y_onehot, do_grad, training)
            if do_grad:
                grads[key], fcs[key] = grad, fc

        logits = F.log_softmax(input, dim=1)

        if do_grad:
            return logits, grads, fcs
        else:
            return logits

    def update_dni_module(self, x, y, y_onehot,
                          task_loss, optimizer, grad_optimizer):
        # forward with self.params
        logits, grads, fcs = self.forward(self.params, x, y_onehot,
                                          do_grad=True, training=True)

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
        loss = task_loss(logits, y)
        loss.backward(retain_graph=True) # need to backward again

        # remove hooks
        for v in handles.values():
            v.remove()

        # dni loss & step
        grad_loss = sum([F.mse_loss(grads[key], real_grads[key].detach())
                         for key in fcs.keys()])

        return loss, grad_loss

