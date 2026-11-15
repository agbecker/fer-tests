import os
# precisa vir ANTES de "import jax": evita que o JAX reserve 75% da VRAM de saída
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from einops import rearrange, repeat

import jax
import jax.numpy as jnp

from cppn import CPPN, FlattenCPPNParameters
from cppn_modified import CPPN as mCPPN, FlattenCPPNParameters as mFlattenCPPNParameters
from color import hsv2rgb
import util


total_size = 1024 * 3
base_size = 256   # mesmo --base_size do treino: passo entre pixels = 2/(base_size-1)
genome = 'apple'
sizes = ["0016", "0032", "0064", "0128", "0256", "0512", "1024"]
sizes = [s for s in sizes if os.path.isdir(f"../varying_sizes/{genome}/n{s}")]

# Mesma grade da célula de erro: base_size + 2*pad = total_size (pad é POR LADO).
pad = (total_size - base_size) // 2
assert base_size + 2 * pad == total_size


def make_renderer(module):
    names = module.inputs.split(",")

    @jax.jit
    def render(p, xs, ys):
        # mesma convenção do generate_image: meshgrid 'ij', eixo 0 = x
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        f = dict(x=X, y=Y, d=jnp.sqrt(X**2 + Y**2) * 1.4, b=jnp.ones_like(X),
                 xabs=jnp.abs(X), yabs=jnp.abs(Y))
        inp = jnp.stack([f[k] for k in names], -1).reshape(-1, len(names))
        (h, s, v), _ = jax.vmap(partial(module.apply, p))(inp)  # features descartadas pelo XLA
        r, g, b = hsv2rgb((h + 1) % 1, s.clip(0, 1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], -1).reshape(X.shape + (3,))
        return (jnp.clip(rgb, 0, 1) * 255).astype(jnp.uint8)

    return render


def gen(fcppn, params, rows_per_chunk=32):
    """Gera a imagem em blocos de linhas para não estourar a VRAM.
    rows_per_chunk deve dividir total_size (evita recompilar no último bloco)."""
    p = fcppn.param_reshaper.reshape_single(params)
    render = make_renderer(fcppn.cppn)
    step = 2.0 / (base_size - 1)
    coords = jnp.linspace(-1.0 - pad * step, 1.0 + pad * step, total_size)
    out = np.empty((total_size, total_size, 3), dtype=np.uint8)
    for i in range(0, total_size, rows_per_chunk):
        xs = coords[i:i + rows_per_chunk]
        out[i:i + len(xs)] = np.asarray(render(p, xs, coords))  # traz pro host a cada bloco
    return out


plt.figure(figsize=(4 * len(sizes), 5), dpi=75)
for iplt, size in enumerate(sizes, start=1):
    plt.subplot(1, len(sizes), iplt)
    n = int(size)

    save_dir = f"../varying_sizes/{genome}/n{size}"
    arch_ = util.load_pkl(save_dir, "arch")
    params = util.load_pkl(save_dir, "params")
    cppn = mFlattenCPPNParameters(mCPPN(arch_))
    img = gen(cppn, params)  # já sai em uint8

    plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.title(f"{genome.capitalize()} n={n}", fontsize=18)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# salva a figura inteira (todos os tamanhos lado a lado)
destination = f"../varying_sizes/{genome}/global_comparison.png"
plt.savefig(destination, dpi=150, bbox_inches="tight")
plt.show()
