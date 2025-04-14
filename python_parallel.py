from jax import numpy as jnp
# from utils import lobatto, tensor_product_quadrature
from sympy import symbols, Matrix, sin, cos, pi, exp, sqrt, diff, simplify
# from kernels import MaternC6Kernel
from sympy2jax import SymbolicModule as to_jax
import jax
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import equinox as eqx
import jax.random as jr
import time

#### manufacturing a soln
x,y,z = symbols('x y z')

u_true_ = sin(pi * x) * sin(pi * y) * sin(pi * z)
u_lapl_ = diff(u_true_, x, 2) +  diff(u_true_, y, 2) +  diff(u_true_, z, 2)

u_true = lambda x,y,z: to_jax(u_true_)(x=x,y=y,z=z)
f = lambda x,y,z: to_jax(u_lapl_)(x=x,y=y,z=z)

### config
ndims = 3


k = 6
n_1d = (2**k + 1) + 2

boundary_face_vals = g = [10,20,15,50,33,75]
x_init = 0


### dirchlet bcs
x = jnp.zeros((n_1d,)*3)
x = x.at[:,:,0].set(boundary_face_vals[0])
x = x.at[:,:,-1].set(boundary_face_vals[1])
x = x.at[0,:,:].set(boundary_face_vals[2])
x = x.at[-1,:,:].set(boundary_face_vals[3])
x = x.at[:,0,:].set(boundary_face_vals[4])
x = x.at[:,-1,:].set(boundary_face_vals[5])
u_pred = x


c,d = 0,1 ## unit cube
grid = jnp.linspace(c,d,n_1d)
h = grid[1] - grid[0]
grid = jnp.asarray(jnp.meshgrid(*(grid,)*ndims)).reshape(ndims,-1).T

rhs = b = f(grid[:,0], grid[:,1], grid[:,2]).reshape((n_1d,)*ndims)
u_soln = u_true(grid[:,0], grid[:,1], grid[:,2]).reshape((n_1d,)*ndims)

stencil = jnp.zeros((3,)*ndims)
stencil = stencil.at[1,1,0].set(1) ### front/back
stencil = stencil.at[1,1,-1].set(1)
stencil = stencil.at[1,0,1].set(1) ## side/side
stencil = stencil.at[1,-1,1].set(1)
stencil = stencil.at[0,1,1].set(1) ## top/bottom
stencil = stencil.at[-1,1,1].set(1) 



jacobi_stencil = stencil
lapl_stencil = stencil.at[1,1,1].set(-6)


key = jr.PRNGKey(0)
conv_jacobi = eqx.tree_at(lambda k: k.weight, 
                eqx.nn.Conv(3,1,1,3,use_bias=False,key=key), 
                jacobi_stencil[None,None])

conv_lapl = eqx.tree_at(lambda k: k.weight, 
                eqx.nn.Conv(3,1,1,3,use_bias=False,key=key), 
                lapl_stencil[None,None])

# Define trilinear kernel (normalized so weights sum to 1 for 3D linear interp)
kernel = jnp.zeros((3, 3, 3))
kernel = kernel.at[1, 1, 1].set(1.0 / 8)  # center
neighbors = [
    (0, 1, 1), (2, 1, 1),  # x
    (1, 0, 1), (1, 2, 1),  # y
    (1, 1, 0), (1, 1, 2)   # z
]
for i, j, k in neighbors:
    kernel = kernel.at[i, j, k].set(1.0 / 8)

kernel = kernel[None, None, :, :, :]  # (out_chan, in_chan, D, H, W)

up_conv = eqx.nn.Conv3d(1,1,3,use_bias=False,padding=1,key=key)
up_conv = eqx.tree_at(lambda m: m.weight, up_conv, kernel)


def jacobi_nstep(nsteps, x_init, rhs):

    def step(carry, x):
        u = carry
        u = u.at[1:-1, 1:-1, 1:-1].set(conv_jacobi(u[None]).squeeze())
        u = (u - h**2 * rhs) / 6
        return u,None
    
    u,_ = jax.lax.scan(step, x_init, None, length=nsteps)
    return u

def calc_residual(u, rhs):
    Ax = u.at[1:-1, 1:-1, 1:-1].set(conv_lapl(u[None]).squeeze())
    return rhs-Ax

def upsample(u):
    n = u.shape[0]
    f = jnp.zeros((1, 2*n-1, 2*n-1, 2*n-1))
    f = f.at[:, ::2, ::2, ::2].set(u[None, ...])  # inject coarse into even location
    u_upsampled = up_conv(f).squeeze()
    return u_upsampled


#### make u current coarse, doing u[coarsen] is the same as doing u[::2,::2,::2]
interior = (slice(1, -1),) * 3
coarsen = (slice(None, None, 2),) * 3


u_interior = u_pred[interior]
rhs_interior = rhs[interior]





############# done setup ################################################

@jax.jit
def do_multigrid_step(u_interior):
    r_interior = calc_residual(u_interior, rhs_interior)

    r_coarse = r_interior
    rhs_coarse = rhs_interior

    solve_steps_in_stage = [4,8,16,32]
    for steps in solve_steps_in_stage:

        r_coarse = r_coarse[coarsen]
        rhs_coarse = rhs_coarse[coarsen]

        r_coarse = jacobi_nstep(steps, r_coarse, rhs_coarse)

    r_up = r_coarse
    rhs_up = rhs_coarse
    for steps in solve_steps_in_stage[::-1]:
        r_up = upsample(r_up)
        rhs_up = upsample(rhs_up)
        r_up = jacobi_nstep(steps, r_up, rhs_up)


    u_interior = u_interior - r_up 
    return u_interior


for _ in range(100):
    tik = time.perf_counter()
    u_interior = do_multigrid_step(u_interior)
    u_interior.block_until_ready()
    tok = time.perf_counter()
    print(jnp.linalg.norm(u_interior.flatten() - u_soln[interior].flatten()) / jnp.linalg.norm(u_soln[interior].flatten()), tok-tik)
