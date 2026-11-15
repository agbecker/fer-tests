import os
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from cppn import CPPN, FlattenCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="", help="architecture")
group = parser.add_argument_group("data")
group.add_argument("--img_file", type=str, default=None, help="path of image file")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def plot_losses(train_losses, val_losses, path, block=100):
    """Salva o gráfico das perdas de treino e validação.

    train_losses tem uma entrada por iteração; val_losses tem uma por bloco
    de `block` iterações, então o eixo x da validação é ajustado de acordo.
    """
    train_steps = np.arange(1, len(train_losses) + 1)
    val_steps = block * np.arange(1, len(val_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_losses, label="treino (centro)", alpha=0.6)
    ax.plot(val_steps, val_losses, label="validação (moldura)", linewidth=2)
    ax.set_xlabel("iteração")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Perda de treino e validação")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    print(args)

    if args.arch is not None and args.arch.endswith(".pkl"):
        import pickle
        with open(args.arch, "rb") as f:
            args.arch = pickle.load(f)

    PAD = 128
    IMG_SIZE = 256

    # Imagem alvo completa (256 + 2*PAD)
    target_img_full = jnp.array(plt.imread(args.img_file)[:, :, :3])

    # Centro da imagem (região de treino)
    target_img_train = target_img_full[
        PAD:-PAD,
        PAD:-PAD
    ]

    cppn = FlattenCPPNParameters(
        CPPN(args.arch, init_scale=args.init_scale)
    )

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    def train_loss_fn(params):
        img = cppn.generate_image(
            params,
            img_size=IMG_SIZE,
            pad_pixels=PAD
        )

        pred = img[PAD:-PAD, PAD:-PAD]

        return jnp.mean((pred - target_img_train) ** 2)

    def validation_loss_fn(params):
        img = cppn.generate_image(
            params,
            img_size=IMG_SIZE,
            pad_pixels=PAD
        )

        loss = 0.0
        n = 0

        # topo
        diff = img[:PAD] - target_img_full[:PAD]
        loss += jnp.sum(diff**2)
        n += diff.size

        # baixo
        diff = img[-PAD:] - target_img_full[-PAD:]
        loss += jnp.sum(diff**2)
        n += diff.size

        # esquerda
        diff = (
            img[PAD:-PAD, :PAD]
            - target_img_full[PAD:-PAD, :PAD]
        )
        loss += jnp.sum(diff**2)
        n += diff.size

        # direita
        diff = (
            img[PAD:-PAD, -PAD:]
            - target_img_full[PAD:-PAD, -PAD:]
        )
        loss += jnp.sum(diff**2)
        n += diff.size

        return loss / n

    @jax.jit
    def train_step(state, _):
        loss, grad = jax.value_and_grad(train_loss_fn)(state.params)

        grad = grad / jnp.linalg.norm(grad)

        state = state.apply_gradients(grads=grad)

        return state, loss

    val_loss_fn = jax.jit(validation_loss_fn)

    tx = optax.adam(learning_rate=args.lr)
    state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx
    )

    gen_img_fn = jax.jit(
        partial(
            cppn.generate_image,
            img_size=IMG_SIZE,
            pad_pixels=PAD
        )
    )

    train_losses = []
    val_losses = []

    imgs_train = [gen_img_fn(state.params)]

    pbar = tqdm(range(args.n_iters // 100))

    for _ in pbar:

        state, train_loss = jax.lax.scan(
            train_step,
            state,
            None,
            length=100
        )

        val_loss = val_loss_fn(state.params)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar.set_postfix(
            train=f"{train_loss.mean():.6f}",
            val=f"{val_loss:.6f}"
        )

        imgs_train.append(gen_img_fn(state.params))

    train_losses = np.array(jnp.concatenate(train_losses))
    val_losses = np.array(val_losses)

    imgs_train = np.array(jnp.stack(imgs_train))

    params = state.params
    img = gen_img_fn(params)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

        # Configuração e arquitetura
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", args.arch)

        # Pesos da rede treinada (pickle + .npy, já que são um vetor único)
        util.save_pkl(args.save_dir, "params", params)
        np.save(os.path.join(args.save_dir, "params.npy"), np.array(params))

        # Curvas de perda (dados brutos) e gráfico
        util.save_pkl(args.save_dir, "train_losses", train_losses)
        util.save_pkl(args.save_dir, "val_losses", val_losses)
        plot_losses(
            train_losses,
            val_losses,
            os.path.join(args.save_dir, "losses.png"),
        )

        # Imagem gerada (região de treino + extrapolação)
        plt.imsave(os.path.join(args.save_dir, "img.png"), np.array(img))


if __name__ == '__main__':
    main(parse_args())