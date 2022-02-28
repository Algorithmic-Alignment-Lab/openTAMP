import torch

class TorchFunc():
    def __init__(self, dim):
        self.input_tensor = torch.zeros(dim, requires_grad=True)
    

    def eval_f(self, x):
        self.input_tensor[:] = x
        return self._eval_f(self.input_tensor).item()


    def _eval_f(self, x):
        raise NotImplementedError()


    def eval_grad(self, x):
        return self._eval_grad(x).item()


    def _eval_grad(self, x):
        val = self._eval_f(x)
        val.backward()
        return self.input_tensor.grad


    def eval_hessian(self, x):
        return self._eval_hessian(x).item()


    def _eval_hessian(self, x):
        self.input_tensor[:] = x
        return F.hessian(self._eval_f, self.input_tensor)


class ThetaDir(TorchFunc):
    def __init__(self
                 use_forward=True, 
                 use_backward=True):
        super().__init__(6)
        self.use_forward = use_forward 
        self.use_backward = use_backward
   

    def _eval_f(self, x):
        pos1, pos2 = self.input_tensor[:2], self.input_tensor[3:5]
        theta = self.input_tensor[2]
        theta_disp = pos2 - pos1
        theta_dist = torch.norm(theta_disp)
        targ_xpos = -theta_dist * torch.sin(theta)
        targ_ypos = theta_dist * torch.cos(theta)

        if self.use_forward and self.use_backward:
            theta_for = torch.sum((theta_disp - [targ_xpos, targ_ypos])**2)
            theta_opp = torch.sum((theta_disp + [targ_xpos, targ_ypos])**2)
            theta_off = torch.min(theta_for, theta_opp)
        elif self.use_forward:
            torch_off = torch.sum((theta_disp - [targ_xpos, targ_ypos])**2)
        else:
            torch_off = torch.sum((theta_disp + [targ_xpos, targ_ypos])**2)

        return torch_off


class GaussianBump(TorchFunc):
    def __init__(self, radius, dim):
        super().__init__(2*dim)
        self.radius = radus

    
    def _eval_f(self, x):
        eta = 1e-7
        bump_dist_sq = torch.sum((pos1-pos2)**2)
        # Dummy call to invoke identity but preserve grad flow to input
        if self.radius**2 <= bump_dist_sq-eta: return 0 * bump_dist_sq
        return torch.exp(-1. * self.radius**2 / (self.radius**2 - bump_dist_sq))


