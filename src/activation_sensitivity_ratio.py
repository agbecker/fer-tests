import os
# precisa vir ANTES de "import jax": evita que o JAX reserve 75% da VRAM de saída
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import jax.numpy as jnp

from activation_sensitivity import load_cppn, neighbor_offsets, compute_features, _sensitivity
from color import hsv2rgb


def sensitivity_ratio(fcppn, params, radius=1.0, metric="euclidean", img_size=256,
                      extrapolate=True, eps=1e-2, chunk=16384):
    """
    Sensibilidade dos neurônios ocultos dividida pela sensibilidade da imagem RGB final:
        R(p) = S_ocultos(p) / (S_saida(p) + eps * média(S_saida))
    O eps relativo evita explodir nas regiões onde a imagem é constante (S_saida ~ 0).
    A saída é comparada em RGB (não nos neurônios h, s, v crus) porque o clip de s, o abs de v
    e o h irrelevante quando s=0 escondem variações da saída que não aparecem na imagem.

    Retorna (S_ocultos, S_saida, R, R_global), com R_global = média(S_ocultos) / média(S_saida):
    quanto de variação interna a rede tem além do que a imagem exige.
    """
    offsets = neighbor_offsets(radius, metric)
    r = int(np.abs(offsets).max())

    if extrapolate:
        F = compute_features(fcppn, params, img_size, pad_pixels=r, chunk=chunk)
        valid = jnp.ones(F.shape[:2], F.dtype)
    else:
        F = compute_features(fcppn, params, img_size, pad_pixels=0, chunk=chunk)
        F = jnp.pad(F, ((r, r), (r, r), (0, 0)))
        valid = jnp.pad(jnp.ones((img_size, img_size), F.dtype), r)

    offsets = jnp.asarray(offsets)
    s_hidden = _sensitivity(F[..., :-3], valid, offsets, r, img_size)  # últimas 3 features = saída (h, s, v)
    h, s, v = F[..., -3], F[..., -2], F[..., -1]
    rgb = jnp.stack(hsv2rgb((h + 1) % 1, s.clip(0, 1), jnp.abs(v).clip(0, 1)), axis=-1)  # igual ao generate_image
    s_out = _sensitivity(rgb, valid, offsets, r, img_size)

    ratio = s_hidden / (s_out + eps * s_out.mean())
    ratio_global = s_hidden.mean() / s_out.mean()
    return s_hidden, s_out, ratio, ratio_global


def _log_imshow(ax, fig, m, title):
    m = np.maximum(np.asarray(m), 1e-12)
    im = ax.imshow(m, cmap="inferno", norm=LogNorm(vmin=m.min(), vmax=m.max()))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_sensitivity_ratio(img, s_hidden, s_out, ratio, ratio_global, title="", save_path=None):
    """Imagem | S ocultos | S saída | razão, todos os mapas em escala log."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(np.clip(np.asarray(img), 0, 1))
    axes[0].set_title("imagem")
    _log_imshow(axes[1], fig, s_hidden, f"S ocultos (log)\nmédia = {float(s_hidden.mean()):.3e}")
    _log_imshow(axes[2], fig, s_out, f"S imagem RGB (log)\nmédia = {float(s_out.mean()):.3e}")
    _log_imshow(axes[3], fig, ratio, f"S ocultos / S imagem (log)\nrazão global = {float(ratio_global):.3e}")

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
    parser.add_argument("--eps", type=float, default=1e-2, help="regularizador relativo à média de S_saida")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    fcppn, params = load_cppn(args.load_dir)
    s_hidden, s_out, ratio, ratio_global = sensitivity_ratio(
        fcppn, params, args.radius, args.metric, args.img_size,
        extrapolate=not args.no_extrapolate, eps=args.eps)
    img = fcppn.generate_image(params, img_size=args.img_size)

    save_path = args.save_path or f"{args.load_dir}/sensitivity_ratio_{args.metric}_r{args.radius:g}.png"
    plot_sensitivity_ratio(img, s_hidden, s_out, ratio, ratio_global,
                           title=f"{args.load_dir}  |  {args.metric}, r={args.radius:g}", save_path=save_path)
    print(f"S ocultos médio: {float(s_hidden.mean()):.6e}")
    print(f"S imagem médio:  {float(s_out.mean()):.6e}")
    print(f"razão global:    {float(ratio_global):.6e}")
    print(f"salvo em {save_path}")
