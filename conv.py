
import torch
from conv_triton import conv1d_triton, conv3d_triton
from functools import partial
from sympy import symbols, Matrix, sin, cos, pi, exp, sqrt, diff, simplify
from sympytorch import SymPyModule
import numpy as np
import time

dtype = torch.float32
device = 'cuda:0'


n_1d = 129
ndims = 3


x,y,z = symbols('x y z')

u_true_ = sin(pi * x) * sin(pi * y) * sin(pi * z)
u_lapl_ = diff(u_true_, x, 2) +  diff(u_true_, y, 2) +  diff(u_true_, z, 2)
u_true = lambda x,y,z: SymPyModule(expressions=[u_true_])(x=x,y=y,z=z)
f = lambda x,y,z: -1 * SymPyModule(expressions=[u_lapl_])(x=x,y=y,z=z)


c,d = 0,1
grid_1d = torch.linspace(c,d,n_1d)

print(f'{grid_1d.shape=}')
h = grid_1d[1] - grid_1d[0]

meshg = torch.meshgrid(*(grid_1d,)*ndims)
grid = torch.vstack(meshg).reshape(ndims,-1).T

rhs = b = f(grid[:,0], grid[:,1], grid[:,2]).reshape((n_1d,)*ndims).to(device)
u_soln = u_true(grid[:,0], grid[:,1], grid[:,2]).reshape((n_1d,)*ndims)
u_soln = u_soln.to(device)
u_pred = torch.zeros((n_1d,)*3).to(device)

jacobi_stencil = torch.zeros((3,3,3))
jacobi_stencil[1,1,0] = 1
jacobi_stencil[1,1,-1] = 1
jacobi_stencil[1,0,1] = 1
jacobi_stencil[1,-1,1] = 1
jacobi_stencil[0,1,1] = 1
jacobi_stencil[-1,1,1] = 1
jacobi_stencil = jacobi_stencil.to(device)

conv_jacobi = lambda x: conv3d_triton(input=x, kernel=jacobi_stencil)

lapl_stencil = jacobi_stencil.clone()
lapl_stencil[1,1,1] = -6
lapl_stencil = lapl_stencil.to(device)

conv_lapl = lambda x: conv3d_triton(input=x, kernel=lapl_stencil)


kernel = torch.Tensor([0.5,1.,0.5])
kernel = kernel.to(device)
upsample_conv = lambda x: conv1d_triton(input=x, kernel=kernel)

def same_conv1d(x, kernel):
    # x and kernel must be 1D tensors
    x = x.squeeze()
    kernel = kernel.flip(0)  # flip for convolution (not cross-correlation)
    x = x.unsqueeze(0).unsqueeze(0)      # shape (1, 1, L)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1, 1, K)

    padding = kernel.shape[-1] // 2
    out = torch.nn.functional.conv1d(x, kernel, padding=padding)
    return out.squeeze(0).squeeze(0)

upsample_conv = lambda x: same_conv1d(x, kernel=kernel)


calc_error = lambda u_pred: torch.linalg.norm(u_pred.flatten() - u_soln.flatten()) / torch.linalg.norm(u_soln.flatten())

def jacobi_nstep(nsteps, u_init, rhs, h):
    for step in range(nsteps):
        u_init[1:-1,1:-1,1:-1] = (conv_jacobi(u_init) + (h**2 * rhs[1:-1,1:-1,1:-1])) / 6
    return u_init

def calc_residual(u, rhs):
    Ax = torch.zeros_like(rhs)
    Ax[1:-1,1:-1,1:-1] = conv_lapl(u)
    return rhs - Ax


def upsample(u):
    n = u.shape[0]
    f = torch.zeros((2*n-1, 2*n-1, 2*n-1))
    f = f.to(device)

    f[::2, ::2, ::2] = u

    f = torch.vmap(torch.vmap(upsample_conv, in_dims=0), in_dims=0)(f)
    f = f.permute(1,2,0)
    f = torch.vmap(torch.vmap(upsample_conv, in_dims=0), in_dims=0)(f)
    f = f.permute(1,2,0)
    f = torch.vmap(torch.vmap(upsample_conv, in_dims=0), in_dims=0)(f)
    f = f.permute(1,2,0)

    return f


#### make u current coarse, doing u[coarsen] is the same as doing u[::2,::2,::2]
interior = (slice(1, -1),) * 3
coarsen = (slice(None, None, 2),) * 3


############# done setup ################################################


def do_multigrid_step(u_pred, h):
    
    ### smooth before starting
    u_pred = jacobi_nstep(3, u_pred, rhs, h)

    residual = calc_residual(u_pred, rhs)
    e = torch.zeros_like(residual)
    r = residual
    solve_steps_in_stage = [2,32,64,128,1000]

    for steps in solve_steps_in_stage:

        h = h*2
        e = e[coarsen]
        r = r[coarsen]
        e = jacobi_nstep(steps, e, r, h)

    for i,steps in enumerate(solve_steps_in_stage[::-1]):
        
        h = h/2
        e = upsample(e)
        r = upsample(r)
        e = jacobi_nstep(steps, e, r, h)
    
    u_pred = u_pred + e
    u_pred = jacobi_nstep(3, u_pred, rhs, h)

    return u_pred


u_pred = jacobi_nstep(1000, u_pred, rhs, h)
print(u_pred)
e1 = calc_error(u_pred).item()
print(e1)
for _ in range(4):
    tik = time.perf_counter()
    e0 = calc_error(u_pred).item()
    u_pred = do_multigrid_step(u_pred, h)
    tok = time.perf_counter()
    e1 = calc_error(u_pred).item()
    print(f'{e0=:.3f}, {e1=:.3f}, {tok-tik}')

u_pred = jacobi_nstep(1000, u_pred, rhs, h)
e1 = calc_error(u_pred).item()
print(e1)



### raw conv kernel comparison

time_conv3d = lambda: conv3d_triton(u_pred, lapl_stencil)


for i in range(10):
    tik = time.perf_counter()
    x = time_conv3d()
    torch.cuda.synchronize()
    tok = time.perf_counter()
    print(tok-tik)
