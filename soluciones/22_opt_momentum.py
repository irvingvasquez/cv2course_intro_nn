# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

"""Solución de la práctica 22: descenso por gradiente con momento.

Lee los hiperparámetros de un YAML, corre la simulación sobre Beale y
reproduce la trayectoria en una animación (ventana y/o video).
Uso:
    python 22_opt_momentum.py
    python 22_opt_momentum.py 22_opt_momentum.yaml
"""

import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nni


class MomentumOptimizer(nni.GradientDescentOptimizer):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, beta=0.9):
        super().__init__(learning_rate=learning_rate, max_iterations=max_iterations, tolerance=tolerance)
        self.beta = beta
        self.name = "Momentum"
        self.momentum = np.zeros(2)

    def next_params(self, gradient, w):
        self.momentum = [
            self.beta * m + (1 - self.beta) * grad
            for m, grad in zip(self.momentum, gradient)
        ]
        return [
            wi - self.learning_rate * mom
            for wi, mom in zip(w, self.momentum)
        ]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path, maybe_path):
    path = Path(maybe_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def square_view(xs, ys, margin=0.20):
    """Ventana cuadrada que cubre los puntos con al menos `margin` a cada lado."""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    dx = x_max - x_min
    dy = y_max - y_min
    span = max(dx, dy, 1e-6)
    pad_x = margin * (dx if dx > 0 else span)
    pad_y = margin * (dy if dy > 0 else span)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y
    side = max(x_max - x_min, y_max - y_min)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = side / 2.0
    return cx - half, cx + half, cy - half, cy + half


def beale_field(problem, limits, n=200):
    min_w1, max_w1, min_w2, max_w2 = limits
    w1 = np.linspace(min_w1, max_w1, n)
    w2 = np.linspace(min_w2, max_w2, n)
    W1, W2 = np.meshgrid(w1, w2)
    Z_log = np.log(problem.function(W1, W2) + 1)
    return W1, W2, Z_log


def trajectory(optimizer):
    history = optimizer.get_history()
    w1_values = [log["params"][0] for log in history]
    w2_values = [log["params"][1] for log in history]
    return w1_values, w2_values


def view_from_start_and_goal(problem, optimizer, margin=0.20):
    w1_values, w2_values = trajectory(optimizer)
    return square_view(
        [w1_values[0], problem.optimal_params[0]],
        [w2_values[0], problem.optimal_params[1]],
        margin=margin,
    )


def apply_square_axes(ax, limits):
    min_x, max_x, min_y, max_y = limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")


def save_plot(problem, optimizer, plot_path):
    w1_values, w2_values = trajectory(optimizer)
    limits = view_from_start_and_goal(problem, optimizer)
    W1, W2, Z_log = beale_field(problem, limits)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(W1, W2, Z_log, levels=50, cmap="viridis")
    ax.plot(w1_values[0], w2_values[0], "wo", markeredgecolor="k", label="inicio")
    ax.plot(problem.optimal_params[0], problem.optimal_params[1], "ro", label="objetivo")
    ax.plot(w1_values, w2_values, "w.-", markersize=6)
    apply_square_axes(ax, limits)
    ax.set_title(
        f"{optimizer.name}, lr={optimizer.learning_rate}, "
        f"beta={optimizer.beta}, iter={len(w1_values)}"
    )
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def make_animation(problem, optimizer, interval):
    w1_values, w2_values = trajectory(optimizer)
    limits = view_from_start_and_goal(problem, optimizer)
    W1, W2, Z_log = beale_field(problem, limits)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(W1, W2, Z_log, levels=50, cmap="viridis")
    ax.plot(w1_values[0], w2_values[0], "wo", markeredgecolor="k")
    ax.plot(problem.optimal_params[0], problem.optimal_params[1], "ro")
    (line,) = ax.plot([], [], "w.-", markersize=6)
    apply_square_axes(ax, limits)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    title = ax.set_title("")

    def update(frame):
        line.set_data(w1_values[: frame + 1], w2_values[: frame + 1])
        title.set_text(
            f"{optimizer.name}, lr={optimizer.learning_rate}, "
            f"beta={optimizer.beta}, iter {frame + 1}/{len(w1_values)}"
        )
        return line, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(w1_values),
        interval=interval,
        blit=True,
        repeat=False,
    )
    return fig, anim


def main(config_path):
    cfg = load_config(config_path)
    problem = nni.ReducedBealeProblem()
    w_inicial = list(cfg.get("w_inicial", cfg.get("initial_params")))
    interval = cfg.get("interval", 30)
    show_animation = bool(cfg.get("show_animation", True)) and bool(os.environ.get("DISPLAY"))
    save_animation = bool(cfg.get("save_animation", False))

    max_iterations_cap = int(cfg.get("max_iterations_cap", 2000))
    max_iterations_run = min(int(cfg["max_iterations"]), max_iterations_cap)
    if int(cfg["max_iterations"]) > max_iterations_cap:
        print(f"Aviso: max_iterations se recortó a {max_iterations_cap}.")

    optimizer = MomentumOptimizer(
        learning_rate=cfg["learning_rate"],
        max_iterations=max_iterations_run,
        tolerance=cfg["tolerance"],
        beta=cfg["beta"],
    )
    w = optimizer.optimize(
        problem.function,
        lambda w1, w2: [problem.gradient_x(w1, w2), problem.gradient_y(w1, w2)],
        w_inicial,
    )

    history = optimizer.get_history()
    threshold = float(cfg.get("threshold", 0.1))
    k = min(len(history), max_iterations_cap)
    w_eval = np.array(history[k - 1]["params"], dtype=float)
    w_optimo = np.array(problem.optimal_params, dtype=float)
    distancia = float(np.linalg.norm(w_eval - w_optimo))
    llego = np.isfinite(distancia) and distancia < threshold

    print(f"Configuración: {config_path}")
    print(f"w inicial: {w_inicial}")
    print(f"w evaluado: {w_eval}")
    print(f"Mejor valor encontrado: {problem.function(*w_eval)}")
    print(f"Iteraciones: {k} (máximo permitido = {max_iterations_cap})")
    print(f"w óptimo: {w_optimo}")
    print(f"Distancia al óptimo: {distancia:.4f} (umbral = {threshold})")
    print("Evaluación: SÍ se acerca al óptimo." if llego else "Evaluación: NO se acerca al óptimo.")

    if cfg.get("save_plot", False):
        plot_path = resolve_path(config_path, cfg["plot_path"])
        save_plot(problem, optimizer, plot_path)
        print(f"Gráfica: {plot_path}")

    if show_animation:
        fig, anim = make_animation(problem, optimizer, interval)
        print("Mostrando animación. Cierra la ventana para continuar.")
        plt.show()

    if save_animation:
        fig, anim = make_animation(problem, optimizer, interval)
        animation_path = resolve_path(config_path, cfg["animation_path"])
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Guardando video: {animation_path}")
        anim.save(animation_path, writer="ffmpeg", fps=max(1, int(1000 / interval)))
        plt.close(fig)
        print(f"Video: {animation_path}")


if __name__ == "__main__":
    default_yaml = Path(__file__).with_suffix(".yaml")
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_yaml
    main(config_path.resolve())
