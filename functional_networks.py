import torch.nn.functional as F
from .utils import conv_params, linear_params, bnparams, bnstats, \
        flatten_params, flatten_stats, batch_norm


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
                'fc1': linear_params(input_dim*input_size, n_hidden),
                'fc2': linear_params(n_hidden, n_hidden),
                'fc3': linear_params(n_hidden, n_classes)}
            if do_bn:
                params.update({'bn1': bnparams(n_hidden),
                               'bn2': bnparams(n_hidden)})
            return params

        def gen_dni():
            dni = {
                'fc1': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI),
                'fc2': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI),
                'fc3': dni_class(n_classes, n_classes, conditioned=conditioned_DNI)}
            if do_bn:
                dni.update({'bn1': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI),
                            'bn2': dni_class(n_hidden, n_classes, conditioned=conditioned_DNI)})
            return dni

        def gen_stats():
            stats = {}
            if do_bn:
                stats.update({'bn1': bnstats(n_hidden),
                              'bn2': bnstats(n_hidden)})
            return stats

        self.params = flatten_params(gen_params(), device)
        self.stats = flatten_stats(gen_stats(), device)
        self.dni = gen_dni()
        self.grads = {} # need to be cleaned before every forward pass

    def f_fc1(self, input, label=None, training=True):
        fc1 = F.relu(F.linear(input, self.params['fc1.weight'], self.params['fc1.bias']))
        if label is not None:
            self.grads['fc1'] = dni['fc1'](fc1, label)
        if self.do_bn:
            fc1 = batch_norm(fc1, self.params, self.stats, 'bn1', training)
            if label is not None:
                self.grads['bn1'] = dni['bn1'](fc1, label)
        return fc1

    def f_fc2(self, input, label, training=True):
        fc2 = F.relu(F.linear(input, self.params['fc2.weight'], self.params['fc2.bias']))
        if label is not None:
            self.grads['fc2'] = dni['fc2'](fc2, label)
        if self.do_bn:
            fc2 = batch_norm(fc2, self.params, self.stats, 'bn2', training)
            if label is not None:
                self.grads['bn2'] = dni['bn2'](fc2, label)
        return fc2

    def f_fc3(self, input, label, training=True):
        fc3 = F.linear(input, self.params['fc3.weight'], self.params['fc3.bias'])
        if label is not None:
            self.grads['fc3'] = dni['fc3'](fc3, label)
        logit = F.log_softmax(fc3, dim=1)
        return logit

    def f(self, input, params, stats, training=True, dni=None, label=None):
        do_dni = True if label is not None and dni is not None else False

        x = input.view(-1, self.input_dim*self.input_size)
        # fc1
        fc1 = self.f_fc1(x, label, training)
        # fc2
        fc2 = self.f_fc2(fc1, label, training)
        # fc3
        logit = self.f_fc3(fc2, label, training)

        return logit


