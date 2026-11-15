"""Mapas de erro (Picbreeder vs. SGD) das redes treinadas por varying_sizes(_reg).py.

Sempre com o mesmo passo entre pixels do treino (2/(base_size-1)), comparando com a CPPN do Picbreeder:
  --mode local  : para cada n, imagem 3n x 3n (centro n x n = treino, margem de largura n = teste)
  --mode global : todas as redes na MESMA grade total_size x total_size (padrão 3 * maior n);
                  centro n x n = treino, todo o resto = teste

Salva:
  - <out_dir>/<genome>_<mode>_error_maps.png : os mapas de erro lado a lado
  - <out_dir>/<genome>_<mode>_errors.json    : MSE de treino (centro) e de teste (margem) por n
  - <out_dir>/<genome>_<mode>_n<NNNN>_error.npy (opcional, --save_maps): o mapa de erro bruto

Exemplos (rodando de dentro de src/):
  python varying_sizes_error.py --genome apple
  python varying_sizes_error.py --genome apple --mode global
  python varying_sizes_error.py --genome apple --mode global --total_size 1536
  python varying_sizes_error.py --genome skull --sizes 16 32 64 --runs_dir ../data/regularization/l2/skull
"""
import argparse
import glob
import json
import os
import re
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")  # servidor sem display
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from color import hsv2rgb
from cppn_modified import CPPN as mCPPN, FlattenCPPNParameters as mFlattenCPPNParameters
import util

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")

parser = argparse.ArgumentParser()
parser.add_argument("--genome", type=str, required=True, help="skull, butterfly, apple, ...")
parser.add_argument("--mode", type=str, default="local", choices=["local", "global"],
                    help="local: imagem 3n x 3n por rede; global: grade comum total_size x total_size")
parser.add_argument("--total_size", type=int, default=None,
                    help="só no modo global: lado da grade comum (par). Padrão: 3 * maior n")
parser.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="valores de n. Padrão: todas as pastas n* encontradas em --runs_dir")
parser.add_argument("--base_size", type=int, default=256,
                    help="mesmo --base_size do treino: passo entre pixels = 2/(base_size-1)")
parser.add_argument("--runs_dir", type=str, default=None,
                    help="pasta com as subpastas nNNNN. Padrão: ../data/varying_sizes/<genome>")
parser.add_argument("--orig_dir", type=str, default=None,
                    help="pasta da CPPN original. Padrão: ../data/picbreeder_<genome>")
parser.add_argument("--params_name", type=str, default="params",
                    help="nome do .pkl dos parâmetros em cada nNNNN (ex.: params, params_best)")
parser.add_argument("--out_dir", type=str, default=None,
                    help="onde salvar os resultados. Padrão: --runs_dir")
parser.add_argument("--chunk", type=int, default=1 << 20,
                    help="nº máximo de pixels avaliados de uma vez na GPU (limita o pico de memória)")
parser.add_argument("--save_maps", action="store_true", help="salva também cada mapa de erro em .npy")
parser.add_argument("--dpi", type=int, default=75)


def make_inputs(input_names, size, step):
    """Entradas (size*size, d_in) de uma janela size x size centrada na origem, com passo `step`.

    Equivale ao generate_image(img_size=base_size, pad_pixels=...) seguido de recorte central,
    mas gera direto a janela pedida (sem recorte) para qualquer size.
    """
    coords = (jnp.arange(size) - (size - 1) / 2.0) * step
    inputs = {}
    inputs["x"], inputs["y"] = jnp.meshgrid(coords, coords, indexing="ij")
    inputs["d"] = jnp.sqrt(inputs["x"] ** 2 + inputs["y"] ** 2) * 1.4
    inputs["b"] = jnp.ones_like(inputs["x"])
    inputs["xabs"], inputs["yabs"] = jnp.abs(inputs["x"]), jnp.abs(inputs["y"])
    inputs = jnp.stack([inputs[k] for k in input_names.split(",")], axis=-1)
    return inputs.reshape(-1, inputs.shape[-1])


@partial(jax.jit, static_argnums=(0, 3))
def render(flat_cppn, params, inputs, chunk):
    """Cores RGB (N, 3) da CPPN em `inputs` (N, d_in), em lotes de `chunk` pixels."""
    cppn = flat_cppn.cppn
    p = flat_cppn.param_reshaper.reshape_single(params)

    def pixel(x):
        (h, s, v), _ = cppn.apply(p, x)  # as features não usadas são eliminadas pelo XLA
        return jnp.stack([h, s, v])

    if inputs.shape[0] > chunk:
        hsv = jax.lax.map(pixel, inputs, batch_size=chunk)
    else:
        hsv = jax.vmap(pixel)(inputs)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    r, g, b = hsv2rgb((h + 1) % 1, s.clip(0, 1), jnp.abs(v).clip(0, 1))
    return jnp.stack([r, g, b], axis=-1)


@partial(jax.jit, static_argnums=(2, 3))
def error_stats(img_orig, img_sgd, n, off):
    """Mapa de erro (total, total) = MSE por pixel nos canais RGB, e MSE do centro n x n
    (começando em `off`) e da margem (todo o resto)."""
    error_map = ((img_orig - img_sgd) ** 2).mean(axis=-1)
    total = error_map.shape[0]
    center = jnp.zeros((total, total), dtype=bool).at[off:off + n, off:off + n].set(True)
    train_mse = jnp.sum(jnp.where(center, error_map, 0.0)) / (n * n)
    margin_mse = jnp.sum(jnp.where(center, 0.0, error_map)) / (total * total - n * n)
    return error_map, train_mse, margin_mse


def generate(flat_cppn, params, size, step, chunk):
    inputs = make_inputs(flat_cppn.cppn.inputs, size, step)
    return render(flat_cppn, jnp.asarray(params), inputs, chunk).reshape(size, size, 3)


def main(args):
    genome = args.genome
    runs_dir = args.runs_dir or os.path.join(DATA_DIR, "varying_sizes", genome)
    orig_dir = args.orig_dir or os.path.join(DATA_DIR, f"picbreeder_{genome}")
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
    if args.mode == "global":
        max_total = args.total_size or 3 * max(sizes)
        assert max_total % 2 == 0 and all(n % 2 == 0 for n in sizes), \
            "total_size e os n precisam ser pares para o centro n x n ficar alinhado"
        assert max_total >= max(sizes), "total_size menor que o maior n"
    else:
        max_total = 3 * max(sizes)
    print(f"genoma: {genome} | modo: {args.mode} | tamanhos: {sizes} | grade máx.: {max_total} "
          f"| runs: {runs_dir}")

    # CPPN original: gera uma vez no maior tamanho e recorta o centro quando preciso
    # (as paridades batem porque todos os tamanhos são pares).
    arch_orig = util.load_pkl(orig_dir, "arch")
    params_orig = util.load_pkl(orig_dir, "params")
    cppn_orig = mFlattenCPPNParameters(mCPPN(arch_orig))
    img_orig_max = generate(cppn_orig, params_orig, max_total, step, args.chunk)

    fig, axs = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 5), dpi=args.dpi, squeeze=False)
    results = []
    cppn_cache = {}

    for ax, n in zip(axs[0], sizes):
        t0 = time.time()
        total = 3 * n if args.mode == "local" else max_total
        off_n = (total - n) // 2  # canto superior esquerdo do centro de treino n x n
        run_dir = os.path.join(runs_dir, f"n{n:04d}")
        arch_sgd = util.load_pkl(run_dir, "arch")
        params_sgd = util.load_pkl(run_dir, args.params_name)
        if arch_sgd not in cppn_cache:  # reaproveita a compilação quando a arquitetura se repete
            cppn_cache[arch_sgd] = mFlattenCPPNParameters(mCPPN(arch_sgd))
        cppn_sgd = cppn_cache[arch_sgd]

        img_sgd = generate(cppn_sgd, params_sgd, total, step, args.chunk)
        off = (max_total - total) // 2
        img_orig = img_orig_max[off:off + total, off:off + total]

        error_map, train_mse, margin_mse = error_stats(img_orig, img_sgd, n, off_n)
        error_map, train_mse, margin_mse = jax.device_get((error_map, train_mse, margin_mse))
        results.append(dict(n=n, img_size=total, train_mse=float(train_mse), margin_mse=float(margin_mse)))
        print(f"  n={n:>5} | treino (centro) = {train_mse:.6f} | teste (margem) = {margin_mse:.6f} "
              f"| {time.time() - t0:.1f}s")

        if args.save_maps:
            np.save(os.path.join(out_dir, f"{genome}_{args.mode}_n{n:04d}_error.npy"), error_map.astype(np.float32))

        plt.sca(ax)
        im = plt.imshow(error_map, cmap="inferno")
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.title(f"{genome.capitalize()} n={n}", fontsize=18)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.add_patch(plt.Rectangle((off_n - 0.5, off_n - 0.5), n, n, fill=False,
                                   edgecolor="cyan", linewidth=2, linestyle="--"))
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)
        del error_map, img_sgd

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{genome}_{args.mode}_error_maps.png")
    fig.savefig(fig_path)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"{genome}_{args.mode}_errors.json")
    with open(json_path, "w") as f:
        json.dump(dict(genome=genome, mode=args.mode, grid_size=max_total if args.mode == "global" else None,
                       base_size=args.base_size, params_name=args.params_name,
                       runs_dir=os.path.abspath(runs_dir), results=results), f, indent=2)

    grid = "imagem completa 3n x 3n (padding = n)" if args.mode == "local" else \
        f"grade comum {max_total} x {max_total}"
    print(f"\nMSE por rede, {grid} — {genome}:")
    print(f"  {'n':>6} | {'treino (centro)':>16} | {'teste (margem)':>16}")
    for r in results:
        print(f"  {r['n']:>6} | {r['train_mse']:16.6f} | {r['margin_mse']:16.6f}")
    print(f"\nsalvo: {fig_path}\n       {json_path}")


if __name__ == "__main__":
    main(parser.parse_args())
