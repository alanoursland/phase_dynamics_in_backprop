import torch
from torch.optim import Optimizer

class Vadam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.5, adaptive_clip=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay, adaptive_clip=adaptive_clip)
        super(Vadam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, prev_grad = state['exp_avg'], state['exp_avg_sq'], state['prev_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Detect gradient sign changes to reduce momentum when oscillating
                sign_change = (grad * prev_grad) < 0
                exp_avg.mul_(beta1 * (1 - sign_change.float() * group['momentum_decay'])).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Update the previous gradient for next step
                prev_grad.copy_(grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Apply adaptive gradient clipping
                clipped_grad = torch.clamp(exp_avg, min=-group['adaptive_clip'], max=group['adaptive_clip'])

                step_size = group['lr'] * (1 - beta2 ** state['step']) ** 0.5 / (1 - beta1 ** state['step'])

                p.data.addcdiv_(clipped_grad, denom, value=-step_size)

        return loss

# Example usage with a simple model
if __name__ == '__main__':
    model = torch.nn.Linear(10, 1)
    optimizer = Vadam(model.parameters(), lr=0.001, momentum_decay=0.5, adaptive_clip=0.1)
    criterion = torch.nn.MSELoss()

    for _ in range(100):
        optimizer.zero_grad()
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')
