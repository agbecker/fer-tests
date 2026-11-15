"""Saídas das redes treinadas por varying_sizes(_reg).py, para todos os tamanhos de um genoma.

Gera duas figuras (todos os n lado a lado), com o mesmo passo entre pixels do treino (2/(base_size-1)):
  - <out_dir>/<genome>_local_outputs.png  : cada rede na sua imagem 3n x 3n (centro n x n = treino)
  - <out_dir>/<genome>_global_outputs.png : todas as redes na MESMA grade total_size x total_size
                                            (padrão 3 * maior n)

Exemplos (rodando de dentro de src/):
  python varying_sizes_visualize.py --genome apple
  python varying_sizes_visualize.py --genome skull --total_size 1536 --box
  python varying_sizes_visualize.py --genome skull --runs_dir ../data/regularization/l2/skull
"""
import argparse
import glob
import os
import re
import time

import matplotlib
matplotlib.use("Agg")  # servidor sem display
import matplotlib.pyplot as plt
import numpy as np

import jax

from cppn_modified import CPPN as mCPPN, FlattenCPPNParameters as mFlattenCPPNParameters
from varying_sizes_error import DATA_DIR, generate
import util

parser = argparse.ArgumentParser()
parser.add_argument("--genome", type=str, required=True, help="skull, butterfly, apple, ...")
parser.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="valores de n. Padrão: todas as pastas n* encontradas em --runs_dir")
parser.add_argument("--base_size", type=int, default=256,
                    help="mesmo --base_size do treino: passo entre pixels = 2/(base_size-1)")
parser.add_argument("--total_size", type=int, default=None,
                    help="lado da grade comum da figura global (par). Padrão: 3 * maior n")
parser.add_argument("--runs_dir", type=str, default=None,
                    help="pasta com as subpastas nNNNN. Padrão: ../data/varying_sizes/<genome>")
parser.add_argument("--params_name", type=str, default="params",
                    help="nome do .pkl dos parâmetros em cada nNNNN (ex.: params, params_best)")
parser.add_argument("--out_dir", type=str, default=None,
                    help="onde salvar as figuras. Padrão: --runs_dir")
parser.add_argument("--chunk", type=int, default=1 << 20,
                    help="nº máximo de pixels avaliados de uma vez na GPU (limita o pico de memória)")
parser.add_argument("--box", action="store_true",
                    help="desenha o contorno da região de treino n x n em cada painel")
parser.add_argument("--dpi", type=int, default=150)


def to_uint8(img):
    # uint8 ocupa 4x menos que float32 (importa nas grades grandes, ex.: 3072 x 3072)
    return (np.clip(np.asarray(img), 0.0, 1.0) * 255).astype(np.uint8)


def plot_row(panels, genome, path, dpi, box):
    """panels: lista de (n, imagem uint8, canto do centro de treino)."""
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5), dpi=dpi, squeeze=False)
    for ax, (n, img, off) in zip(axs[0], panels):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{genome.capitalize()} n={n}", fontsize=18)
        if box:
            ax.add_patch(plt.Rectangle((off - 0.5, off - 0.5), n, n, fill=False,
                                       edgecolor="cyan", linewidth=1.5, linestyle="--"))
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    genome = args.genome
    runs_dir = args.runs_dir or os.path.join(DATA_DIR, "varying_sizes", genome)
    out_dir = args.out_dir or runs_dir
    os.makedirs(out_dir, exist_ok=True)
    step = 2.0 / (args.base_size - 1)

    print("dispositivos JAX:", jax.devices())
    if jax.default_backend() != "gpu":
        print("AVISO: JAX não está usando GPU.")

    if args.sizes is None:
        found = [re.fullmatch(r"n(\d+)", os.path.basename(p)) for p in glob.glob(os.path.join(runs_dir, "n*"))]
        sizes = sorted(int(m.group(1)) for m in found if m)
    else:
        sizes = sorted(args.sizes)
    sizes = [n for n in sizes if os.path.isfile(os.path.join(runs_dir, f"n{n:04d}", f"{args.params_name}.pkl"))]
    if not sizes:
        raise SystemExit(f"nenhuma pasta nNNNN com {args.params_name}.pkl em {runs_dir}")

    total_global = args.total_size or 3 * max(sizes)
    assert total_global % 2 == 0 and all(n % 2 == 0 for n in sizes), \
        "total_size e os n precisam ser pares para o centro n x n ficar alinhado"
    assert total_global >= max(sizes), "total_size menor que o maior n"
    print(f"genoma: {genome} | tamanhos: {sizes} | grade global: {total_global} | runs: {runs_dir}")

    local_panels, global_panels = [], []
    cppn_cache = {}
    for n in sizes:
        t0 = time.time()
        run_dir = os.path.join(runs_dir, f"n{n:04d}")
        arch = util.load_pkl(run_dir, "arch")
        params = util.load_pkl(run_dir, args.params_name)
        if arch not in cppn_cache:  # reaproveita a compilação quando a arquitetura se repete
            cppn_cache[arch] = mFlattenCPPNParameters(mCPPN(arch))
        cppn = cppn_cache[arch]

        # Global: gera a grade comum. Local (3n x 3n): se couber na grade global, é só um
        # recorte central dela (mesmas coordenadas); senão, gera separado.
        img_global = to_uint8(generate(cppn, params, total_global, step, args.chunk))
        total_local = 3 * n
        if total_local <= total_global:
            off = (total_global - total_local) // 2
            img_local = img_global[off:off + total_local, off:off + total_local].copy()
        else:
            img_local = to_uint8(generate(cppn, params, total_local, step, args.chunk))

        local_panels.append((n, img_local, n))
        global_panels.append((n, img_global, (total_global - n) // 2))
        print(f"  n={n:>5} | local {total_local}x{total_local} | global {total_global}x{total_global} "
              f"| {time.time() - t0:.1f}s")

    local_path = os.path.join(out_dir, f"{genome}_local_outputs.png")
    global_path = os.path.join(out_dir, f"{genome}_global_outputs.png")
    plot_row(local_panels, genome, local_path, args.dpi, args.box)
    plot_row(global_panels, genome, global_path, args.dpi, args.box)
    print(f"\nsalvo: {local_path}\n       {global_path}")


if __name__ == "__main__":
    main(parser.parse_args())
