import numpy as np
import torch
import torch.autograd.functional as F


class TorchFunc():
    def __init__(self, dim, device='cpu'):
        self.device = device
    

    def eval_f(self, x):
        x = torch.tensor(x.flatten()).to(self.device)
        return self._eval_f(x).item()


    def _eval_f(self, x):
        raise NotImplementedError()


    def eval_grad(self, x):
        x = torch.tensor(x.flatten(), requires_grad=True).to(self.device)
        return self._eval_grad(x).numpy()


    def _eval_grad(self, x):
        val = self._eval_f(x)
        val.backward()
        return x.grad


    def eval_hess(self, x):
        return self._eval_hess(x).numpy()


    def _eval_hess(self, x):
        x = torch.tensor(x.flatten(), requires_grad=True).to(self.device)
        return F.hessian(self._eval_f, x)


class ThetaDir(TorchFunc):
    def __init__(self,
                 use_forward=True, 
                 use_reverse=True):
        super().__init__(6)
        self.use_forward = use_forward 
        self.use_reverse = use_reverse
   

    def _eval_f(self, x):
        pos1, pos2 = x[:2], x[3:5]
        theta = x[2]
        theta_disp = pos2 - pos1
        theta_dist = torch.norm(theta_disp)
        targ_xpos = -theta_dist * torch.sin(theta)
        targ_ypos = theta_dist * torch.cos(theta)
        targ_disp = torch.tensor([targ_xpos, targ_ypos])

        if self.use_forward and self.use_reverse:
            theta_for = (theta_disp[0] - targ_xpos)**2 + (theta_disp[1] - targ_ypos)**2
            theta_opp = (theta_disp[0] + targ_xpos)**2 + (theta_disp[1] + targ_ypos)**2
            theta_off = torch.min(theta_for, theta_opp)

        elif self.use_forward:
            theta_off = (theta_disp[0] - targ_xpos)**2 + (theta_disp[1] - targ_ypos)**2

        else:
            theta_off = (theta_disp[0] + targ_xpos)**2 + (theta_disp[1] + targ_ypos)**2

        return theta_off


class GaussianBump(TorchFunc):
    def __init__(self, radius, dim, scale=1., eta=1e-4):
        super().__init__(2*dim)
        self.radius = radius
        self.scale = np.e * scale # Bump height is 1/e, so we scale to 1 first
        self.eta = eta
    

    def _eval_f(self, x):
        pos1, pos2 = x[:2], x[2:]
        bump_dist_sq = torch.sum((pos1 - pos2)**2)
        # Dummy call to invoke identity but preserve grad flow to input
        if (self.radius-self.eta)**2 <= bump_dist_sq: return 0. * bump_dist_sq
        return self.scale * torch.exp(-1. * self.radius**2 / (self.radius**2 - bump_dist_sq))


