import torch.nn.functional as F
import torch.nn as nn
from utils import conv_params, linear_params, bnparams, bnstats, batch_norm
import itertools
from dni import dni_linear, dni_Conv2d

class FUNCTIONAL_NET:
    def __init__(self, device='cpu', conditioned_DNI=False):
        self.device = device
        self.conditioned_DNI = conditioned_DNI
        self.params = {}

    def init_theta(self, key):
        #return {k:v.detach().requires_grad_() for k,v in self.params[key].items()}
        return {k:v.clone() for k,v in self.params[key].items()}

    def preprocess(self, x):
        return x


class MLP_DNI_FCx3(FUNCTIONAL_NET):
    def __init__(self, input_dim=1, input_size=28*28, device='cpu', do_bn=False,
                 n_hidden=400, n_classes=10, conditioned_DNI=False):
        super().__init__(device, conditioned_DNI)

        self.input_dim = input_dim
        self.input_size = input_size
        self.do_bn = do_bn
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        def gen_params():
            params = {
                'fc1': linear_params(input_dim*input_size, n_hidden, device),
                'fc2': linear_params(n_hidden, n_hidden, device),
                'fc3': linear_params(n_hidden, n_classes, device)}
            functions = {'fc1': F.linear, 'fc2': F.linear, 'fc3': F.linear}
            activations = {'fc1': F.relu, 'fc2': F.relu, 'fc3': None}
            return params, functions, activations

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
                'fc1': dni_linear(n_hidden, n_classes,
                                 dni_hidden_size=n_hidden, conditioned=conditioned_DNI).to(device),
                'fc2': dni_linear(n_hidden, n_classes,
                                 dni_hidden_size=n_hidden, conditioned=conditioned_DNI).to(device),
                'fc3': dni_linear(n_classes, n_classes,
                                 dni_hidden_size=n_hidden, conditioned=conditioned_DNI).to(device)}
            dni_seq = nn.Sequential(dni['fc1'], dni['fc2'], dni['fc3'])
            return dni, dni_seq

        # All params
        self.params, self.functions, self.activations = gen_params()
        if do_bn:
            self.bn_params = gen_bn_params()
            self.bn_stats = gen_bn_stats()
        self.dni, self.dni_seq = gen_dni()

    def parameters(self):
        for lyr in self.params.values():
            for w in lyr.values():
                yield w
        if self.do_bn:
            for lyr in self.bn_params.values():
                for w in lyr.values():
                    yield w

    def dni_parameters(self):
        return itertools.chain(self.dni['fc1'].parameters(),
                               self.dni['fc2'].parameters(),
                               self.dni['fc3'].parameters())

    def layer(self, key, theta, input, y_onehot=None, do_grad=False, training=True):
        '''
        FC-ReLU-BN OR FC
        '''
        # linear
        fc = self.functions[key](input, theta['weight'], theta['bias'])

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

    def preprocess(self, x):
        return x.view(-1, self.input_dim*self.input_size)

    def forward(self, params, x, y_onehot=None, do_grad=False, training=True):
        '''
        forward without theta refinement
        '''
        grads, fcs = {}, {}

        input = self.preprocess(x)

        for key in self.params.keys():
            input, grad, fc = self.layer(key, params[key], input, y_onehot, do_grad, training)
            if do_grad:
                grads[key], fcs[key] = grad, fc

        logits = F.log_softmax(input, dim=1) # TODO: support regression

        if do_grad:
            return logits, grads, fcs
        else:
            return logits


class CNN_DNI_CONVx2_FCx1(FUNCTIONAL_NET):
    def __init__(self, input_dim=1, input_size=None, device='cpu', do_bn=False,
                 n_hidden=16, n_classes=10, conditioned_DNI=False):
        super().__init__(device, conditioned_DNI)

        self.input_dim = input_dim
        self.do_bn = do_bn
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        def gen_params():
            params = {
                'conv1': conv_params(input_dim, n_hidden, k=5, device),
                'conv2': conv_params(n_hidden, n_hidden*2, k=5, device),
                'fc1': linear_params(7*7*n_hidden*2, n_classes, device)}
            functions = {'conv1': F.conv2d, 'conv2': F.conv2d, 'fc1': F.linear}
            args = {'conv1': {'padding':2}, 'conv2': {'padding':2}, 'fc1': {}}
            activations = {'conv1': F.relu, 'conv2': F.relu, 'fc1': None}
            return params, functions, args, activations

        def gen_bn_params():
            params = {'conv1': bnparams(n_hidden, device),
                      'conv2': bnparams(n_hidden*2, device)}
            return params

        def gen_bn_stats():
            stats = {'conv1': bnstats(n_hidden, device),
                     'conv2': bnstats(n_hidden*2, device)}
            return stats

        def gen_dni():
            dni = {
                'conv1': dni_Conv2d(n_hidden, (14, 14), n_classes,
                                    dni_hidden_size=n_hidden*2, conditioned=conditioned_DNI).to(device),
                'conv2': dni_Conv2d(n_hidden, (7, 7), n_classes,
                                    dni_hidden_size=n_hidden*2, conditioned=conditioned_DNI).to(device),
                'fc1': dni_linear(n_classes, n_classes,
                                  dni_hidden_size=n_hidden*2, conditioned=conditioned_DNI).to(device)}
            dni_seq = nn.Sequential(dni['conv1'], dni['conv2'], dni['fc1'])
            return dni, dni_seq

        # All params
        self.params, self.functions, self.args, self.activations = gen_params()
        if do_bn:
            self.bn_params = gen_bn_params()
            self.bn_stats = gen_bn_stats()
        self.dni, self.dni_seq = gen_dni()

    def parameters(self):
        for lyr in self.params.values():
            for w in lyr.values():
                yield w
        if self.do_bn:
            for lyr in self.bn_params.values():
                for w in lyr.values():
                    yield w

    def dni_parameters(self):
        return itertools.chain(self.dni['conv1'].parameters(),
                               self.dni['conv2'].parameters(),
                               self.dni['fc1'].parameters())

    def layer(self, key, theta, input, y_onehot=None, do_grad=False, training=True):
        '''
        Conv2D-ReLU-BN-MaxPool2d OR FC
        '''
        # linear or Conv2d
        if self.functions[key] == F.linear:
            input = input.view(input.size(0), -1)
        fc = self.functions[key](input, theta['weight'], theta['bias'], **self.args[key])

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

        # downsampling
        if self.activations[key] is not None:
            output = F.max_pool2d(output, 2)

        return output, grad, fc

    def forward(self, params, input, y_onehot=None, do_grad=False, training=True):
        '''
        forward without theta refinement
        '''
        grads, fcs = {}, {}

        for key in self.params.keys():
            input, grad, fc = self.layer(key, params[key], input, y_onehot, do_grad, training)
            if do_grad:
                grads[key], fcs[key] = grad, fc

        logits = F.log_softmax(input, dim=1)

        if do_grad:
            return logits, grads, fcs
        else:
            return logits

