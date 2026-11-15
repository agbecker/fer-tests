"""
Treina, via SGD, uma CPPN com a arquitetura REDUZIDA da CPPN do Picbreeder de um genoma.

Reduzida = a arquitetura original ("N;act:n,...") sem os neurônios zerados, isto é,
neurônios ocultos que não recebem nenhuma entrada (todos os pesos de entrada vindos de
neurônios vivos são zero) ou que não produzem saída (todos os pesos de saída para
neurônios vivos são zero). A remoção é repetida até estabilizar, já que tirar um neurônio
pode zerar outro. O número de camadas é mantido; cada camada fica com seu próprio
número de neurônios, no formato por camada "cache:6|cache:5,sin:2|...".

Uso:
    python cppn_minimal.py --genome skull --save_dir ../data/sgd_minimal_skull

Os hiperparâmetros (seed, lr, n_iters) são os mesmos usados em train_sgd.py para cada
genoma (ver GENOME_CONFIG).
"""
import os
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import evosax

from color import hsv2rgb
import util

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# mesmos hiperparâmetros dos comandos train_sgd.py do fer.ipynb
GENOME_CONFIG = {
    "skull":     dict(seed=2, lr=3e-3, n_iters=100000),
    "butterfly": dict(seed=0, lr=3e-3, n_iters=100000),
    "apple":     dict(seed=0, lr=3e-4, n_iters=100000),  # precisa de lr menor
}


# ----------------------------------------------------------------------------------------
# Arquitetura por camada
# ----------------------------------------------------------------------------------------

def parse_arch(arch):
    """
    Converte a string de arquitetura em uma lista de camadas; cada camada é uma lista de (ativação, n_neurônios).

    Formatos aceitos:
      - constante (igual ao cppn.py):  "12;cache:15,gaussian:4,identity:2,sin:1"
      - por camada, separado por "|":  "cache:15,gaussian:4|sin:8,tanh:2|cache:4,identity:2"
    Ativações com 0 neurônios são ignoradas.
    """
    if ";" in arch:
        n_layers, spec = arch.split(";")
        layer_specs = [spec] * int(n_layers)
    else:
        layer_specs = arch.split("|")

    layers = []
    for spec in layer_specs:
        layer = []
        for item in spec.split(","):
            activation, n = item.strip().split(":")
            if activation not in activation_fn_map:
                raise ValueError(f"ativação desconhecida: {activation!r}")
            if int(n) > 0:
                layer.append((activation, int(n)))
        if not layer:
            raise ValueError(f"camada sem neurônios em {arch!r}")
        layers.append(layer)
    return layers


def build_arch(layers):
    """Inverso de parse_arch: [[("cache", 15), ("sin", 2)], ...] -> "cache:15,sin:2|..." """
    return "|".join(",".join(f"{a}:{n}" for a, n in layer) for layer in layers)


class CPPN(nn.Module):
    """
    CPPN Flax Model com número de neurônios (e ativações) definido por camada.
    Possible activations: cache (identity), identity, cos, sin, tanh, sigmoid, gaussian, relu.

    arch: str no formato por camada "cache:15,gaussian:4|sin:8,tanh:2|..." (uma entrada por camada oculta),
          ou no formato constante "12;cache:15,gaussian:4,identity:2,sin:1" (compatível com cppn.py).
    inputs: str should be in the form "y,x,d,b" which means the inputs are y, x, d, b. Don't change this.
    init_scale: str should be in the form "default" or float. If default uses the default flax initialization scheme (lecun init). If float, it is the scale of the initialization variance (see code).
    """
    arch: str = "cache:15,gaussian:4,identity:2,sin:1|cache:8,sin:4|cache:4,gaussian:2"
    inputs: str = "y,x,d,b" # "x,y,d,b,xabs,yabs"
    init_scale: str = "default"

    @nn.compact
    def __call__(self, x):
        layers = parse_arch(self.arch)

        features = [x]
        for layer in layers:
            activations = [a for a, _ in layer]
            d_hidden = [n for _, n in layer]
            dh_cumsum = list(np.cumsum(d_hidden))

            if self.init_scale == "default":
                x = nn.Dense(sum(d_hidden), use_bias=False)(x)
            else:
                kernel_init = nn.initializers.variance_scaling(scale=float(self.init_scale), mode="fan_in", distribution="truncated_normal")
                x = nn.Dense(sum(d_hidden), use_bias=False, kernel_init=kernel_init)(x)

            x = jnp.split(x, dh_cumsum)
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, activations)]
            x = jnp.concatenate(x)

            features.append(x)
        x = nn.Dense(3, use_bias=False)(x)
        features.append(x)
        h, s, v = x
        return (h, s, v), features

    def generate_image(self, params, img_size=256, pad_pixels=0, return_features=False):
        """
        Generate an image from the CPPN given the parameters at the resolution specified by img_size.
        If return_features is True, return the intermediate activations of the CPPN as well.

        pad_pixels: number of extra pixels to add on EACH side, beyond the original [-1, 1] domain,
        using the same spacing as the original img_size x img_size grid. Output resolution = img_size + 2 * pad_pixels.
        """
        step = 2.0 / (img_size - 1)  # spacing between pixels in the original grid
        total_size = img_size + 2 * pad_pixels
        coord_min = -1.0 - pad_pixels * step
        coord_max = 1.0 + pad_pixels * step

        inputs = {}
        x = y = jnp.linspace(coord_min, coord_max, total_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['b'] = jnp.ones_like(inputs['x'])
        inputs['xabs'], inputs['yabs'] = jnp.abs(inputs['x']), jnp.abs(inputs['y'])
        inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        inputs = jnp.stack(inputs, axis=-1)
        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)
        if return_features:
            return rgb, features
        else:
            return rgb


class FlattenCPPNParameters():
    """
    Using evosax==0.1.6 (only old version works!!), flatten the parameters of the CPPN to a single vector.
    """
    def __init__(self, cppn):
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        d_in = len(self.cppn.inputs.split(","))
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((d_in,))))
        self.n_params = self.param_reshaper.total_params

    def init(self, rng):
        d_in = len(self.cppn.inputs.split(","))
        params = self.cppn.init(rng, jnp.zeros((d_in,)))
        return self.param_reshaper.flatten_single(params)

    def generate_image(self, params, img_size=256, pad_pixels=0, return_features=False):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(
            params,
            img_size=img_size,
            pad_pixels=pad_pixels,
            return_features=return_features
        )


# ----------------------------------------------------------------------------------------
# Redução da arquitetura
# ----------------------------------------------------------------------------------------

def get_kernels(arch, params):
    """Kernels (d_in, d_out) de cada camada Dense, em ordem, a partir dos params achatados."""
    tree = FlattenCPPNParameters(CPPN(arch)).param_reshaper.reshape_single(params)["params"]
    return [np.array(tree[f"Dense_{i}"]["kernel"]) for i in range(len(tree))]


def find_alive_neurons(kernels):
    """
    Máscara booleana de neurônios vivos para cada camada (entrada, ocultas, saída).
    Entradas e saídas estão sempre vivas. Um neurônio oculto morre se não tem nenhum peso
    não nulo vindo de um neurônio vivo da camada anterior, ou nenhum peso não nulo indo
    para um neurônio vivo da camada seguinte. Repete até estabilizar.
    """
    alive = [np.ones(K.shape[0], dtype=bool) for K in kernels] + [np.ones(kernels[-1].shape[1], dtype=bool)]
    changed = True
    while changed:
        changed = False
        for l in range(1, len(kernels)):
            has_input = (kernels[l - 1][alive[l - 1]] != 0).any(axis=0)
            has_output = (kernels[l][:, alive[l + 1]] != 0).any(axis=1)
            new = alive[l] & has_input & has_output
            if (new != alive[l]).any():
                alive[l] = new
                changed = True
    for l, a in enumerate(alive[1:-1], start=1):
        if not a.any():
            raise ValueError(f"a camada oculta {l} ficou sem neurônios")
    return alive


def reduce_arch(arch, params):
    """
    Remove os neurônios zerados da CPPN (arch, params).

    Retorna (arch_reduzida, params_reduzidos, alive):
      - arch_reduzida: string no formato por camada;
      - params_reduzidos: pesos originais restritos aos neurônios vivos (só para conferência,
        o treino parte de uma inicialização aleatória);
      - alive: máscaras de neurônios vivos por camada.
    A ordem dos neurônios é preservada, então os grupos de ativação continuam contíguos.
    """
    layers = parse_arch(arch)
    kernels = get_kernels(arch, params)
    alive = find_alive_neurons(kernels)

    new_layers = []
    for layer, a in zip(layers, alive[1:-1]):
        acts = np.concatenate([[act] * n for act, n in layer])
        new_layer = [(act, int(a[acts == act].sum())) for act, _ in layer]
        new_layers.append([(act, n) for act, n in new_layer if n > 0])
    arch_reduced = build_arch(new_layers)

    new_kernels = [K[alive[l]][:, alive[l + 1]] for l, K in enumerate(kernels)]
    cppn_reduced = FlattenCPPNParameters(CPPN(arch_reduced))
    tree = {"params": {f"Dense_{i}": {"kernel": jnp.array(K)} for i, K in enumerate(new_kernels)}}
    params_reduced = cppn_reduced.param_reshaper.flatten_single(tree)
    return arch_reduced, params_reduced, alive


# ----------------------------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------------------------

def train(arch, target_img, seed, lr, n_iters, init_scale="default"):
    """Mesmo laço de train_sgd.py: Adam, gradiente normalizado, MSE contra a imagem alvo."""
    cppn = FlattenCPPNParameters(CPPN(arch, init_scale=init_scale))
    params = cppn.init(jax.random.PRNGKey(seed))

    def loss_fn(params):
        img = cppn.generate_image(params, img_size=256)
        return jnp.mean((img - target_img)**2)

    # optax direto em vez de flax TrainState: nas versões novas do flax,
    # TrainState.apply_gradients faz `"..." in grads`, o que quebra com params achatados.
    tx = optax.adam(learning_rate=lr)

    @jax.jit
    def train_step(state, _):
        params, opt_state = state
        loss, grad = jax.value_and_grad(loss_fn)(params)
        grad = grad / jnp.linalg.norm(grad)
        updates, opt_state = tx.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    state = (params, tx.init(params))

    losses = []
    pbar = tqdm(range(n_iters // 100))
    for _ in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        losses.append(loss)
        pbar.set_postfix(loss=loss.mean().item())

    params = state[0]
    losses = np.array(jnp.concatenate(losses))
    img = cppn.generate_image(params, img_size=256)
    return params, losses, img


def main(genome, save_dir, n_iters=None):
    cfg = dict(GENOME_CONFIG[genome])
    if n_iters is not None:
        cfg["n_iters"] = n_iters

    pb_dir = os.path.join(DATA_DIR, f"picbreeder_{genome}")
    arch_orig = util.load_pkl(pb_dir, "arch")
    params_orig = util.load_pkl(pb_dir, "params")
    target_img = jnp.array(plt.imread(os.path.join(pb_dir, "img.png"))[:, :, :3])

    arch, params_pruned, alive = reduce_arch(arch_orig, params_orig)
    n_orig = FlattenCPPNParameters(CPPN(arch_orig)).n_params
    n_red = FlattenCPPNParameters(CPPN(arch)).n_params
    print(f"genoma: {genome}")
    print(f"arch original: {arch_orig} ({n_orig} params)")
    print(f"arch reduzida: {arch} ({n_red} params)")
    print(f"neurônios por camada: {[int(a.sum()) for a in alive[1:-1]]}")

    params, losses, img = train(arch, target_img, **cfg)
    print(f"loss final: {losses[-100:].mean():.6f}")

    os.makedirs(save_dir, exist_ok=True)
    util.save_pkl(save_dir, "args", dict(genome=genome, save_dir=save_dir, arch_orig=arch_orig, **cfg))
    util.save_pkl(save_dir, "arch", arch)
    util.save_pkl(save_dir, "params", params)
    util.save_pkl(save_dir, "losses", losses)
    util.save_pkl(save_dir, "alive", alive)
    util.save_pkl(save_dir, "params_pruned_picbreeder", params_pruned)
    plt.imsave(os.path.join(save_dir, "img.png"), np.array(img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, required=True, choices=list(GENOME_CONFIG))
    parser.add_argument("--save_dir", type=str, required=True, help="pasta onde salvar os resultados")
    args = parser.parse_args()
    main(args.genome, args.save_dir)
