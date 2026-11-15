from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

import evosax

from color import hsv2rgb

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)


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
