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
from jax import random as jr, vmap, jit
import time

jax.config.update("jax_enable_x64", True)
#### manufacturing a soln
x,y,z = symbols('x y z')

u_true_ = sin(pi * x) * sin(pi * y) * sin(pi * z)
u_lapl_ = diff(u_true_, x, 2) +  diff(u_true_, y, 2) +  diff(u_true_, z, 2)
u_true = lambda x,y,z: to_jax(u_true_)(x=x,y=y,z=z)
f = lambda x,y,z: -1 * to_jax(u_lapl_)(x=x,y=y,z=z)

### config
ndims = 3


k = 6
n_1d = (2**k + 1) 

x = jnp.zeros((n_1d,)*3)
u_pred = x


c,d = 0,1
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
                        eqx.nn.Conv(3,1,1,3,
                                    padding='same',
                                    use_bias=False,
                                    key=key),
                                    lapl_stencil[None,None])

# Define trilinear kernel (normalized so weights sum to 1 for 3D linear interp)
kernel = jnp.array([0.5,1,0.5])
fine_conv = lambda x: jnp.convolve(x.squeeze(), kernel, mode='same')
### right hand input is kernel? 

calc_error = lambda u_pred: jnp.linalg.norm(u_pred.flatten() - u_soln.flatten()) / jnp.linalg.norm(u_soln.flatten())

def jacobi_nstep(nsteps, x_init, rhs):

    def step(carry, x):
        u = carry
        u = u.at[1:-1,1:-1,1:-1].set((conv_jacobi(u[None]).squeeze() + (h**2 * rhs[1:-1,1:-1,1:-1])) / 6)
        return u,None
    
    u,_ = jax.lax.scan(step, x_init, None, length=nsteps)
    return u

def calc_residual(u, rhs):
    Ax = conv_lapl(u[None]).squeeze()
    return rhs - Ax


# u_pred = jacobi_nstep(20, u_pred, rhs)
# print(u_pred.shape)

# # for s in range(100):
# #     u_pred = jacobi_nstep(20, u_pred, rhs)
# #     print(calc_error(u_pred))


def upsample(u):
    n = u.shape[0]
    f = jnp.zeros((2*n-1, 2*n-1, 2*n-1))
    f = f.at[::2, ::2, ::2].set(u)  # inject coarse into even location
    kernel = jnp.array([0.5,1,0.5]) / 2.
    fine_conv = lambda x: jnp.convolve(x.squeeze(), kernel, mode='same')

    ### convolve each axis
    f = vmap(vmap(fine_conv, in_axes=0), in_axes=0)(f)
    f = f.transpose(1,2,0)
    f = vmap(vmap(fine_conv, in_axes=0), in_axes=0)(f)
    f = f.transpose(1,2,0)
    f = vmap(vmap(fine_conv, in_axes=0), in_axes=0)(f)
    f = f.transpose(1,2,0)

    return f


#### make u current coarse, doing u[coarsen] is the same as doing u[::2,::2,::2]
interior = (slice(1, -1),) * 3
coarsen = (slice(None, None, 2),) * 3


############# done setup ################################################

@jit
def do_multigrid_step(u_pred):
    
    u_pred = jacobi_nstep(3, u_pred, rhs)
    residual = calc_residual(u_pred, rhs)

    e = jnp.zeros_like(residual)
    r = residual
    
    solve_steps_in_stage = [2,2,2,2,100]
    for steps in solve_steps_in_stage:
        e = e[coarsen]
        r = r[coarsen]
        e = jacobi_nstep(steps, e, r)

    for i,steps in enumerate(solve_steps_in_stage[::-1]):
        e = upsample(e)
        r = upsample(r)
        e = jacobi_nstep(steps, e, r)

    return e


for _ in range(1000):
    tik = time.perf_counter()

    correction = do_multigrid_step(u_pred)
    u_pred = u_pred + correction
    u_pred = jacobi_nstep(3, u_pred, rhs)
    tok = time.perf_counter()
    print(calc_error(u_pred))
