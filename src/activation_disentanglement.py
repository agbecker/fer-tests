import os
# precisa vir ANTES de "import jax": evita que o JAX reserve 75% da VRAM de saída
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.scipy.special import entr

from activation_sensitivity import load_cppn, compute_features


"""
Entanglement e fracture por perturbação de atributo (estilo DCI, com importância por perturbação).

Atributos da CPPN: x e y (d e b são derivados deles). Para cada pixel p, neurônio oculto k e atributo j:
    M_kj(p) = média_{±} (h_k(p ± step·e_j) - h_k(p))²
    (perturbações de ±step pixels, avaliadas pela própria CPPN — inclusive fora de [-1, 1])

Entanglement (um neurônio responde a vários atributos?):
    H_k(p) = entropia_j de M_kj / Σ_j M_kj, normalizada por log(n_atributos)  -> [0, 1]
    E(p)   = média de H_k(p) sobre neurônios, ponderada por Σ_j M_kj (neurônios parados não contam)
Fracture (um atributo está espalhado por vários neurônios?):
    G_j(p) = entropia_k de M_kj / Σ_k M_kj, normalizada por log(n_neurônios)  -> [0, 1]
    Fr(p)  = média de G_j(p) sobre atributos, ponderada por Σ_k M_kj
Medidas globais: as mesmas médias ponderadas, mas também sobre todos os pixels
(regiões onde a rede não muda não pesam).

Rede perfeitamente unificada e desenrolada (um neurônio por atributo) -> E = Fr = 0.
Ressalva: "desenrolado" aqui é relativo aos eixos x e y; uma CPPN que usa d (radial) mistura x e y
por construção, então E > 0 não é necessariamente um defeito nela.
"""


@partial(jax.jit, static_argnames=("step", "img_size"))
def _importance(F, step, img_size):
    """F: (N+2s, N+2s, K) -> M: (N, N, K, 2), último eixo = (x, y). Eixo 0 da grade = x (meshgrid 'ij')."""
    s, n = step, img_size
    ref = F[s:s + n, s:s + n]
    mx = ((F[2 * s:2 * s + n, s:s + n] - ref) ** 2 + (F[:n, s:s + n] - ref) ** 2) / 2
    my = ((F[s:s + n, 2 * s:2 * s + n] - ref) ** 2 + (F[s:s + n, :n] - ref) ** 2) / 2
    return jnp.stack([mx, my], axis=-1)


def importance_matrix(fcppn, params, img_size=256, step=1, include_output=False, chunk=16384):
    """M_kj(p) com shape (img_size, img_size, K, 2). Por padrão só neurônios ocultos (sem entradas nem h, s, v)."""
    F = compute_features(fcppn, params, img_size, pad_pixels=step, chunk=chunk)
    if not include_output:
        F = F[..., :-3]
    return _importance(F, step, img_size)


def _weighted_entropy(M, axis):
    """Entropia normalizada ao longo de `axis`, ponderada pela massa total de cada distribuição.
    Retorna (mapa por pixel, valor global)."""
    tot = M.sum(axis)
    p = M / jnp.maximum(jnp.expand_dims(tot, axis), 1e-30)
    H = entr(p).sum(axis) / jnp.log(M.shape[axis])

    num, den = (tot * H).sum(-1), tot.sum(-1)  # média sobre neurônios (ou atributos) no pixel
    local = jnp.where(den > 0, num / jnp.maximum(den, 1e-30), jnp.nan)
    glob = num.sum() / jnp.maximum(den.sum(), 1e-30)
    return local, glob


@jax.jit
def entanglement_fracture(M):
    """M: (N, N, K, J) -> (E_map, E_global, Fr_map, Fr_global)."""
    e_map, e_glob = _weighted_entropy(M, axis=-1)   # por neurônio, entropia sobre atributos
    f_map, f_glob = _weighted_entropy(M, axis=-2)   # por atributo, entropia sobre neurônios
    return e_map, e_glob, f_map, f_glob


def plot_entanglement_fracture(img, e_map, e_glob, f_map, f_glob, title="", save_path=None):
    """Imagem | mapa de entanglement | mapa de fracture (escala linear em [0, 1])."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(np.clip(np.asarray(img), 0, 1))
    axes[0].set_title("imagem")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")  # pixels onde nenhum neurônio muda
    for ax, m, name, g in [(axes[1], e_map, "entanglement", e_glob), (axes[2], f_map, "fracture", f_glob)]:
        im = ax.imshow(np.asarray(m), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{name}\nglobal = {float(g):.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--step", type=int, default=1, help="tamanho da perturbação, em pixels")
    parser.add_argument("--include_output", action="store_true", help="inclui os neurônios de saída (h, s, v)")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    fcppn, params = load_cppn(args.load_dir)
    M = importance_matrix(fcppn, params, args.img_size, args.step, include_output=args.include_output)
    e_map, e_glob, f_map, f_glob = entanglement_fracture(M)
    img = fcppn.generate_image(params, img_size=args.img_size)

    save_path = args.save_path or f"{args.load_dir}/entanglement_fracture_s{args.step}.png"
    plot_entanglement_fracture(img, e_map, e_glob, f_map, f_glob,
                               title=f"{args.load_dir}  |  step={args.step}", save_path=save_path)
    print(f"entanglement global: {float(e_glob):.4f}")
    print(f"fracture global:     {float(f_glob):.4f}")
    print(f"salvo em {save_path}")
