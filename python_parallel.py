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


k = 7
n_1d = (2**k + 1) 

x = jnp.zeros((n_1d,)*3)
u_pred = x


c,d = 0,1
grid_1d = jnp.linspace(c,d,n_1d)
print(f'{grid_1d.shape=}')
h = grid_1d[1] - grid_1d[0]
grid = jnp.asarray(jnp.meshgrid(*(grid_1d,)*ndims)).reshape(ndims,-1).T


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
                                    use_bias=False,
                                    key=key),
                                    lapl_stencil[None,None])

# Define trilinear kernel (normalized so weights sum to 1 for 3D linear interp)
kernel = jnp.array([0.5,1,0.5])
fine_conv = lambda x: jnp.convolve(x.squeeze(), kernel, mode='same')
### right hand input is kernel? 

calc_error = lambda u_pred: jnp.linalg.norm(u_pred.flatten() - u_soln.flatten()) / jnp.linalg.norm(u_soln.flatten())

def jacobi_nstep(nsteps, u_init, rhs, h):

    def step(carry, x):
        u = carry
        u = u.at[1:-1,1:-1,1:-1].set((conv_jacobi(u[None]).squeeze() + (h**2 * rhs[1:-1,1:-1,1:-1])) / 6)
        return u,None
    
    u,_ = jax.lax.scan(step, u_init, None, length=nsteps)
    return u


def calc_residual(u, rhs):
    Ax = jnp.zeros_like(rhs)
    Ax = Ax.at[1:-1,1:-1,1:-1].set(conv_lapl(u[None]).squeeze())
    return rhs - Ax

def upsample(u):
    n = u.shape[0]
    f = jnp.zeros((2*n-1, 2*n-1, 2*n-1))
    f = f.at[::2, ::2, ::2].set(u)  # inject coarse into even location
    kernel = jnp.array([0.5,1,0.5])
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


# grid_test = jnp.asarray(jnp.meshgrid(*(np.arange(2**2+1),)*3))[0]
# print(grid_test, 'og')
# g_down = grid_test[coarsen]
# g_up = upsample(g_down)
# print(g_up, 'new')
# print()

############# done setup ################################################

@jit
def do_multigrid_step(u_pred, h):
    
    ### smooth before starting
    u_pred = jacobi_nstep(3, u_pred, rhs, h)

    residual = calc_residual(u_pred, rhs)
    e = jnp.zeros_like(residual)
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
e1 = calc_error(u_pred).item()
print(e1)
for _ in range(1):
    tik = time.perf_counter()
    e0 = calc_error(u_pred).item()
    u_pred = do_multigrid_step(u_pred, h)
    u_pred.block_until_ready()
    tok = time.perf_counter()
    e1 = calc_error(u_pred).item()
    print(f'{e0=:.3f}, {e1=:.3f}, {tok-tik}')

u_pred = jacobi_nstep(1000, u_pred, rhs, h)
e1 = calc_error(u_pred).item()
print(e1)
plt.gca().set_aspect('equal', adjustable='box')  # Set the aspect ratio to square
a = plt.contourf(np.linspace(0,1,129),np.linspace(0,1,129),u_pred[64])
# plt.show()
plt.colorbar(a)
plt.savefig('stuff')


time_conv3d = lambda: conv_jacobi(u_pred[None])

import timeit

for i in range(10):
    tik = time.perf_counter()
    x = time_conv3d()
    x.block_until_ready()
    tok = time.perf_counter()
    print(tok-tik)

# 0.000386479077860713
# 1.7579877749085426e-05