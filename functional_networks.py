import torch.nn.functional as F
from .utils import conv_params, linear_params, bnparams, bnstats, batch_norm


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
        for layer in self.dni.values():
            for w in layer.values():
                yield w

    def init_theta(self, key):
        #return {k:v.detach().requires_grad_() for k,v in self.params[key].items()}
        #return {k:v.clone() for k,v in self.params[key].items()}
        return self.params[key]

    def f_fc(self, input, theta, key, label=None, training=True):
        # linear
        fc = F.linear(input, theta['weight'], theta['bias'])

        # dni
        if label is not None:
            grad = self.dni[key](fc, label) # d_loss/d_fc
        else:
            grad = None

        # activation
        output = fc if self.activations[key] is None else self.activations[key](fc)

        # batchnorm
        if self.do_bn:
            output = batch_norm(output, self.bn_params, self.bn_stats, key, training)

        return output, grad

    def forward(self, input, params, label=None, training=True):
        input = input.view(-1, self.input_dim*self.input_size)
        for k in self.params.keys():
            input, _ = self.f_fc(input, params[k], k, label, training)
        logit = F.log_softmax(input, dim=1)

        return logit

