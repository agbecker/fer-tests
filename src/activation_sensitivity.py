import os
# precisa vir ANTES de "import jax": evita que o JAX reserve 75% da VRAM de saída
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import jax
import jax.numpy as jnp

from cppn import CPPN, FlattenCPPNParameters
import util


def load_cppn(load_dir):
    """Carrega arch.pkl + params.pkl de um diretório (sgd_*, picbreeder_*, varying_sizes/...)."""
    arch = util.load_pkl(load_dir, "arch")
    params = jnp.asarray(np.asarray(util.load_pkl(load_dir, "params")))
    return FlattenCPPNParameters(CPPN(arch)), params


def neighbor_offsets(radius, metric="euclidean"):
    """Deslocamentos (dy, dx) com 0 < dist <= radius. metric: "euclidean" ou "manhattan"."""
    r = int(np.floor(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    if metric == "euclidean":
        dist = np.hypot(dy, dx)
    elif metric == "manhattan":
        dist = np.abs(dy) + np.abs(dx)
    else:
        raise ValueError(f"métrica desconhecida: {metric!r}")
    mask = (dist > 0) & (dist <= radius)
    if not mask.any():
        raise ValueError(f"raio {radius} não inclui nenhum vizinho")
    return np.stack([dy[mask], dx[mask]], axis=-1).astype(np.int32)


def compute_features(fcppn, params, img_size=256, pad_pixels=0, include_inputs=False, chunk=16384):
    """
    Ativações de todos os neurônios em cada pixel: array (N, N, D), N = img_size + 2*pad_pixels.
    Mesma grade/convenção do generate_image (meshgrid 'ij', eixo 0 = x).
    include_inputs: inclui (y, x, d, b) no vetor de ativações. Saída (h, s, v) pré-HSV sempre entra.
    chunk: pixels avaliados por vez (lax.map) para limitar o pico de VRAM.
    """
    module = fcppn.cppn
    p = fcppn.param_reshaper.reshape_single(params)
    names = module.inputs.split(",")

    step = 2.0 / (img_size - 1)
    n = img_size + 2 * pad_pixels
    coords = jnp.linspace(-1.0 - pad_pixels * step, 1.0 + pad_pixels * step, n)

    @jax.jit
    def run(p):
        X, Y = jnp.meshgrid(coords, coords, indexing="ij")
        f = dict(x=X, y=Y, d=jnp.sqrt(X**2 + Y**2) * 1.4, b=jnp.ones_like(X),
                 xabs=jnp.abs(X), yabs=jnp.abs(Y))
        inp = jnp.stack([f[k] for k in names], -1).reshape(-1, len(names))

        def pixel(x):
            _, feats = module.apply(p, x)
            return jnp.concatenate(feats if include_inputs else feats[1:])

        F = jax.lax.map(pixel, inp, batch_size=chunk)
        return F.reshape(n, n, -1)

    return run(p)


def _sensitivity(F, valid, offsets, r, img_size):
    """
    F: (N+2r, N+2r, D) ativações; valid: (N+2r, N+2r) 1 onde o vizinho conta.
    Percorre os deslocamentos com lax.scan: só guarda um mapa (N, N) acumulado,
    então a memória é O(N²·D) independente do número de vizinhos.
    """
    D = F.shape[-1]
    ref = jax.lax.dynamic_slice(F, (r, r, 0), (img_size, img_size, D))

    def body(carry, off):
        acc, cnt = carry
        y0, x0 = r + off[0], r + off[1]
        nb = jax.lax.dynamic_slice(F, (y0, x0, 0), (img_size, img_size, D))
        w = jax.lax.dynamic_slice(valid, (y0, x0), (img_size, img_size))
        acc = acc + w * jnp.mean((nb - ref) ** 2, axis=-1)  # média sobre os neurônios
        return (acc, cnt + w), None

    zeros = jnp.zeros((img_size, img_size), F.dtype)
    (acc, cnt), _ = jax.lax.scan(body, (zeros, zeros), offsets)
    return acc / cnt  # média sobre os vizinhos


_sensitivity = jax.jit(_sensitivity, static_argnames=("r", "img_size"))


def activation_sensitivity(fcppn, params, radius=1.0, metric="euclidean", img_size=256,
                           extrapolate=True, include_inputs=False, chunk=16384):
    """
    Mapa (img_size, img_size) de sensibilidade de ativação:
        S(p) = média_{q em N(p)} média_k (f_k(q) - f_k(p))²,   N(p) = {q : 0 < dist(p, q) <= radius}

    extrapolate=True: vizinhos fora da imagem são avaliados pela própria CPPN (ela é definida
        fora de [-1, 1], com o mesmo passo entre pixels), então todo pixel tem vizinhança completa.
    extrapolate=False: vizinhos fora da imagem são ignorados (a média usa só os válidos).
    """
    offsets = neighbor_offsets(radius, metric)
    r = int(np.abs(offsets).max())

    if extrapolate:
        F = compute_features(fcppn, params, img_size, pad_pixels=r, include_inputs=include_inputs, chunk=chunk)
        valid = jnp.ones(F.shape[:2], F.dtype)
    else:
        F = compute_features(fcppn, params, img_size, pad_pixels=0, include_inputs=include_inputs, chunk=chunk)
        F = jnp.pad(F, ((r, r), (r, r), (0, 0)))
        valid = jnp.pad(jnp.ones((img_size, img_size), F.dtype), r)

    return _sensitivity(F, valid, jnp.asarray(offsets), r, img_size)


def plot_sensitivity(img, sens, title="", log=True, save_path=None):
    """Imagem gerada ao lado do mapa de calor da sensibilidade."""
    sens = np.asarray(sens)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(np.clip(np.asarray(img), 0, 1))
    axes[0].set_title("imagem")

    norm = LogNorm(vmin=max(sens[sens > 0].min(), 1e-12), vmax=sens.max()) if log and (sens > 0).any() else None
    im = axes[1].imshow(np.maximum(sens, 1e-12) if norm else sens, cmap="inferno", norm=norm)
    axes[1].set_title("sensibilidade de ativação" + (" (log)" if norm else "") + f"\nmédia = {sens.mean():.3e}")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, required=True, help="diretório com arch.pkl e params.pkl")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--no_extrapolate", action="store_true", help="ignora vizinhos fora da imagem")
    parser.add_argument("--include_inputs", action="store_true", help="inclui as entradas (y, x, d, b)")
    parser.add_argument("--linear", action="store_true", help="escala de cor linear em vez de log")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    fcppn, params = load_cppn(args.load_dir)
    sens = activation_sensitivity(fcppn, params, args.radius, args.metric, args.img_size,
                                  extrapolate=not args.no_extrapolate, include_inputs=args.include_inputs)
    img = fcppn.generate_image(params, img_size=args.img_size)

    save_path = args.save_path or f"{args.load_dir}/sensitivity_{args.metric}_r{args.radius:g}.png"
    plot_sensitivity(img, sens, title=f"{args.load_dir}  |  {args.metric}, r={args.radius:g}",
                     log=not args.linear, save_path=save_path)
    print(f"sensibilidade média: {float(sens.mean()):.6e}")
    print(f"salvo em {save_path}")
