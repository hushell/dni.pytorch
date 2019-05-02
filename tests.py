import torch

##################################################################
lr = 0.01
n_inner = 5

theta = torch.rand(2,1).requires_grad_() # theta_0

psi_mu = torch.rand(2,1).requires_grad_()
psi_rho = torch.rand(2,1).requires_grad_()

def L(t, m, r):
    return (t+m+r).min(), (t+m+r).mean()

for t in range(n_inner):
    loss, grad = L(theta, psi_mu, psi_rho)
    loss.backward(retain_graph=True)
    theta = theta - lr * grad # theta_t

# zero grad
psi_mu.grad.detach_()
psi_mu.grad.zero_()
psi_rho.grad.detach_()
psi_rho.grad.zero_()

loss, grad = L(theta, psi_mu, psi_rho)
loss.backward()

print(psi_mu.grad)

##################################################################
