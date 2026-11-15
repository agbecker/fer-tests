import os
import json
import argparse
from functools import partial

# Por padrão o JAX reserva 75% da memória da GPU logo no início. Desativa isso para
# a memória crescer conforme o necessário (útil ao rodar vários tamanhos em sequência).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import optax

from cppn import CPPN, FlattenCPPNParameters
from color import hsv2rgb
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default="results", help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="", help="architecture (string ou .pkl)")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")

group = parser.add_argument_group("data")
group.add_argument("--target_params", type=str, required=True,
                   help="parâmetros (vetor achatado, .npy ou .pkl) da CPPN que gera o alvo")
group.add_argument("--target_arch", type=str, default=None,
                   help="arquitetura da CPPN alvo (string ou .pkl). Se omitido, usa --arch")
group.add_argument("--name", type=str, default="target",
                   help="prefixo do nome das imagens da CPPN alvo (ex.: skull -> skull16.png, skull64.png)")
group.add_argument("--base_size", type=int, default=256,
                   help="tamanho de referência: define o espaçamento entre pixels (2/(base_size-1)). "
                        "Com base 256, uma imagem de 256 px cobre [-1, 1]; uma de 512 px cobre [-2, 2], etc.")
group.add_argument("--sizes", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024],
                   help="valores de n (imagem final 3n x 3n, treino no centro n x n)")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=20000,
                   help="iterações para o menor tamanho da lista")
group.add_argument("--iters_growth", type=float, default=0.0,
                   help="iters(n) = n_iters * (n / n_min) ** iters_growth. "
                        "0 = número fixo; 0.5 = cresce com a raiz do tamanho; 1 = linear")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--keep", type=str, default="best", choices=["best", "last"],
                   help="salvar a rede de menor loss de treino ou a última")

group = parser.add_argument_group("gpu")
group.add_argument("--block", type=int, default=500,
                   help="iterações por chamada compilada (lax.scan). Blocos maiores = menos "
                        "idas e voltas CPU<->GPU")
group.add_argument("--render_chunk", type=int, default=1 << 20,
                   help="nº máximo de pixels avaliados de uma vez ao gerar as imagens 3n x 3n "
                        "(limita o pico de memória da GPU em tamanhos grandes)")
group.add_argument("--remat", action="store_true",
                   help="recomputa as ativações no backward (jax.checkpoint): menos memória "
                        "da GPU no treino com n grande, ao custo de ~1 forward extra por passo")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


def make_inputs(input_names, size, step):
    """Entradas da CPPN para uma janela size x size com espaçamento fixo `step`, centrada na origem.

    Retorna um array achatado (size*size, d_in). As coordenadas vão de
    -(size-1)/2*step até +(size-1)/2*step. Com step = 2/(256-1) e size = 256
    isso reproduz exatamente linspace(-1, 1, 256) do CPPN.generate_image.
    É calculado uma única vez e passado como argumento às funções jitadas
    (em vez de virar uma constante enorme embutida no programa compilado).
    """
    coords = (jnp.arange(size) - (size - 1) / 2.0) * step
    inputs = {}
    inputs['x'], inputs['y'] = jnp.meshgrid(coords, coords, indexing='ij')
    inputs['d'] = jnp.sqrt(inputs['x'] ** 2 + inputs['y'] ** 2) * 1.4
    inputs['b'] = jnp.ones_like(inputs['x'])
    inputs['xabs'], inputs['yabs'] = jnp.abs(inputs['x']), jnp.abs(inputs['y'])
    inputs = jnp.stack([inputs[k] for k in input_names.split(",")], axis=-1)
    return inputs.reshape(-1, inputs.shape[-1])


def render(flat_cppn, params, inputs, chunk=None, remat=False):
    """Avalia a CPPN em `inputs` (N, d_in) e retorna as cores RGB (N, 3).

    Com `chunk`, os pixels são processados em lotes via lax.map para limitar a
    memória; sem `chunk`, tudo num único vmap (mais rápido quando cabe na GPU).
    """
    cppn = flat_cppn.cppn
    p = flat_cppn.param_reshaper.reshape_single(params)

    def pixel(x):
        (h, s, v), _ = cppn.apply(p, x)  # as features não usadas são eliminadas pelo XLA
        return jnp.stack([h, s, v])

    if remat:
        pixel = jax.checkpoint(pixel)

    if chunk is not None and inputs.shape[0] > chunk:
        hsv = jax.lax.map(pixel, inputs, batch_size=chunk)
    else:
        hsv = jax.vmap(pixel)(inputs)

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    r, g, b = hsv2rgb((h + 1) % 1, s.clip(0, 1), jnp.abs(v).clip(0, 1))
    return jnp.stack([r, g, b], axis=-1)


def load_arch(arch):
    if arch is not None and arch.endswith(".pkl"):
        import pickle
        with open(arch, "rb") as f:
            return pickle.load(f)
    return arch


def load_flat_params(path):
    """Carrega um vetor de parâmetros achatado (formato do FlattenCPPNParameters)."""
    if path.endswith(".npy"):
        return jnp.array(np.load(path))
    import pickle
    with open(path, "rb") as f:
        return jnp.array(np.asarray(pickle.load(f)))


def plot_train_loss(train_losses, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses)
    ax.set_xlabel("iteração")
    ax.set_ylabel("MSE (treino)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_size(args, arch, target_cppn, target_params, n, n_iters):
    out_dir = os.path.join(args.save_dir, f"n{n:04d}")
    os.makedirs(out_dir, exist_ok=True)
    full = 3 * n
    step = 2.0 / (args.base_size - 1)  # espaçamento fixo entre pixels, igual em todos os tamanhos
    train_range = (n - 1) / 2.0 * step
    full_range = (full - 1) / 2.0 * step
    print(f"\n=== n={n} | imagem {full}x{full} | treino {n}x{n} | {n_iters} iterações ===")
    print(f"  domínio de treino: [-{train_range:.3f}, {train_range:.3f}] | "
          f"imagem completa: [-{full_range:.3f}, {full_range:.3f}]")

    cppn = FlattenCPPNParameters(CPPN(arch, init_scale=args.init_scale))
    params = cppn.init(jax.random.PRNGKey(args.seed))

    # Entradas calculadas uma vez e mantidas na GPU.
    # O centro n x n da imagem 3n tem exatamente as mesmas coordenadas do treino.
    full_inputs = make_inputs(target_cppn.cppn.inputs, full, step)
    if cppn.cppn.inputs != target_cppn.cppn.inputs:
        model_full_inputs = make_inputs(cppn.cppn.inputs, full, step)
    else:
        model_full_inputs = full_inputs
    train_inputs = make_inputs(cppn.cppn.inputs, n, step)

    render_full = jax.jit(partial(render, chunk=args.render_chunk), static_argnums=0)

    # Alvo: a CPPN de referência gerando a imagem completa 3n x 3n (fica na GPU).
    target_full = render_full(target_cppn, target_params, full_inputs).reshape(full, full, 3)
    target_train = target_full[n:2 * n, n:2 * n].reshape(-1, 3)

    def train_loss_fn(params, inputs, target):
        img = render(cppn, params, inputs, remat=args.remat)
        return jnp.mean((img - target) ** 2)

    tx = optax.adam(learning_rate=args.lr)

    # Um bloco de `args.block` passos de otimização numa única chamada compilada.
    # O melhor conjunto de parâmetros é acompanhado dentro do scan, passo a passo,
    # sem forward extra e sem sincronizar com a CPU. A loss do passo i é a dos
    # parâmetros antes da atualização i, então best_params corresponde exatamente a best_loss.
    @partial(jax.jit, donate_argnums=(0,))
    def run_block(state, inputs, target):
        def step_fn(state, _):
            params, opt_state, best_loss, best_params = state
            loss, grad = jax.value_and_grad(train_loss_fn)(params, inputs, target)
            better = loss < best_loss
            best_loss = jnp.where(better, loss, best_loss)
            best_params = jnp.where(better, params, best_params)
            grad = grad / (jnp.linalg.norm(grad) + 1e-12)
            updates, opt_state = tx.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, best_loss, best_params), loss
        return jax.lax.scan(step_fn, state, None, length=args.block)

    # best_params precisa ser um buffer próprio (os buffers do estado são doados ao jit)
    state = (params, tx.init(params), jnp.array(jnp.inf, dtype=jnp.float32), jnp.copy(params))

    n_blocks = max(1, n_iters // args.block)
    train_losses = []
    best_seen = float("inf")
    pending = None

    def collect(losses):
        nonlocal best_seen
        losses = np.asarray(losses)
        train_losses.append(losses)
        best_seen = min(best_seen, float(losses.min()))
        pbar.set_postfix(train=f"{losses[-1]:.6f}", best=f"{best_seen:.6f}")

    pbar = tqdm(range(n_blocks))
    for _ in pbar:
        # Despacho assíncrono: enfileira o próximo bloco na GPU *antes* de ler na CPU
        # as losses do bloco anterior, assim a GPU nunca fica ociosa esperando o Python.
        state, losses = run_block(state, train_inputs, target_train)
        if pending is not None:
            collect(pending)
        pending = losses
    collect(pending)
    pbar.close()

    params, _, _, best_params = state
    final_params = best_params if args.keep == "best" else params
    train_losses = np.concatenate(train_losses)

    # Imagem completa 3n x 3n (centro = treino, margens = teste) e métricas, na GPU.
    @jax.jit
    def evaluate(final_params, inputs, target_full):
        img = render(cppn, final_params, inputs, chunk=args.render_chunk).reshape(full, full, 3)
        sq = jnp.sum((img - target_full) ** 2, axis=-1)
        center = jnp.zeros((full, full), dtype=bool).at[n:2 * n, n:2 * n].set(True)
        train_mse = jnp.sum(jnp.where(center, sq, 0.0)) / (3 * n * n)
        test_mse = jnp.sum(jnp.where(center, 0.0, sq)) / (3 * (full * full - n * n))
        return img, train_mse, test_mse

    img, train_mse, test_mse = evaluate(final_params, model_full_inputs, target_full)
    img, target_full, train_mse, test_mse, final_params = jax.device_get(
        (img, target_full, train_mse, test_mse, final_params))

    # Métricas finais (as margens nunca foram vistas no treino)
    metrics = dict(
        n=n, img_size=full, n_iters=n_iters, keep=args.keep,
        base_size=args.base_size, pixel_step=step,
        train_coord_range=train_range, full_coord_range=full_range,
        train_mse=float(train_mse),
        test_mse=float(test_mse),
    )
    print(f"  train MSE={metrics['train_mse']:.6f} | test (margens) MSE={metrics['test_mse']:.6f}")

    # Salvamento: rede e imagem separadas
    util.save_pkl(out_dir, "args", args)
    util.save_pkl(out_dir, "arch", arch)
    util.save_pkl(out_dir, "params", final_params)
    np.save(os.path.join(out_dir, "params.npy"), np.array(final_params))
    util.save_pkl(out_dir, "train_losses", train_losses)
    plot_train_loss(train_losses, os.path.join(out_dir, "train_loss.png"))
    plt.imsave(os.path.join(out_dir, "img.png"), np.clip(img, 0.0, 1.0))
    plt.imsave(os.path.join(out_dir, "target.png"), np.clip(target_full, 0.0, 1.0))
    # Imagem da CPPN alvo também na raiz, com nome do tipo "skull16.png"
    plt.imsave(os.path.join(args.save_dir, f"{args.name}{n}.png"), np.clip(target_full, 0.0, 1.0))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main(args):
    print(args)
    print("dispositivos JAX:", jax.devices())
    if jax.default_backend() != "gpu":
        print("AVISO: JAX não está usando GPU (instale jax[cuda12]).")

    arch = load_arch(args.arch)
    target_arch = load_arch(args.target_arch) if args.target_arch is not None else arch

    target_cppn = FlattenCPPNParameters(CPPN(target_arch))
    target_params = load_flat_params(args.target_params)
    assert target_params.shape == (target_cppn.n_params,), (
        f"target_params tem shape {target_params.shape}, esperado ({target_cppn.n_params},) "
        "para a arquitetura alvo"
    )

    sizes = sorted(args.sizes)
    all_metrics = []
    for n in sizes:
        n_iters = int(round(args.n_iters * (n / sizes[0]) ** args.iters_growth))
        all_metrics.append(run_size(args, arch, target_cppn, target_params, n, n_iters))
        jax.clear_caches()  # libera as compilações do tamanho anterior

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
