# Attribution 4.0 International
# Juan Irving Vasquez
# jivg.org

"""Solución de la práctica 21: descenso por gradiente en dos dimensiones.

Lee los hiperparámetros de un YAML, corre la simulación sobre la esfera y
reproduce la trayectoria en una animación (ventana y/o video).
Uso:
    python 21_opt_gradiente_2d.py
    python 21_opt_gradiente_2d.py 21_opt_gradiente_2d.yaml
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


class GradientDescentOptimizer(nni.Optimizer):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        super().__init__(tolerance=tolerance)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = "Descenso por gradiente"

    def next_params(self, gradient, w):
        return [
            wi - self.learning_rate * grad
            for wi, grad in zip(w, gradient)
        ]

    def optimize(self, function, gradient, w_inicial):
        self.history = []
        w = w_inicial
        iteration = 0

        while iteration < self.max_iterations:
            log = {"iter": iteration, "params": w, "value": function(*w)}
            self.history.append(log)

            gradient_values = gradient(*w)
            w_nuevo = self.next_params(gradient_values, w)
            if self._converged(w, w_nuevo):
                break
            w = w_nuevo
            iteration += 1
        return w

    def _converged(self, w, w_nuevo):
        return all(
            abs(wi - wj) < self.tolerance
            for wi, wj in zip(w, w_nuevo)
        )

    def get_history(self):
        return self.history


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path, maybe_path):
    path = Path(maybe_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def sphere_field(problem, n=100):
    w1 = np.linspace(problem.min_x, problem.max_x, n)
    w2 = np.linspace(problem.min_y, problem.max_y, n)
    W1, W2 = np.meshgrid(w1, w2)
    Z = problem.function(W1, W2)
    return W1, W2, Z


def trajectory(optimizer):
    history = optimizer.get_history()
    w1_values = [log["params"][0] for log in history]
    w2_values = [log["params"][1] for log in history]
    return w1_values, w2_values


def save_plot(problem, optimizer, plot_path):
    w1_values, w2_values = trajectory(optimizer)
    W1, W2, Z = sphere_field(problem)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.contourf(W1, W2, Z, levels=50, cmap="viridis")
    plt.plot(problem.optimal_params[0], problem.optimal_params[1], "ro")
    plt.plot(w1_values, w2_values, "w.-", markersize=6)
    plt.title(
        f"{optimizer.name}, lr={optimizer.learning_rate}, iter={len(w1_values)}"
    )
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_2$")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def make_animation(problem, optimizer, interval):
    w1_values, w2_values = trajectory(optimizer)
    W1, W2, Z = sphere_field(problem)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(W1, W2, Z, levels=50, cmap="viridis")
    ax.plot(problem.optimal_params[0], problem.optimal_params[1], "ro")
    (line,) = ax.plot([], [], "w.-", markersize=6)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    title = ax.set_title("")

    def update(frame):
        line.set_data(w1_values[: frame + 1], w2_values[: frame + 1])
        title.set_text(
            f"{optimizer.name}, lr={optimizer.learning_rate}, "
            f"iter {frame + 1}/{len(w1_values)}"
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
    problem = nni.SphereProblem()
    w_inicial = list(cfg.get("w_inicial", cfg.get("initial_params")))
    interval = cfg.get("interval", 30)
    show_animation = bool(cfg.get("show_animation", True)) and bool(os.environ.get("DISPLAY"))
    save_animation = bool(cfg.get("save_animation", True))

    optimizer = GradientDescentOptimizer(
        learning_rate=cfg["learning_rate"],
        max_iterations=cfg["max_iterations"],
        tolerance=cfg["tolerance"],
    )
    w = optimizer.optimize(
        problem.function,
        lambda w1, w2: [problem.gradient_x(w1, w2), problem.gradient_y(w1, w2)],
        w_inicial,
    )

    print(f"Configuración: {config_path}")
    print(f"w inicial: {w_inicial}")
    print(f"w: {w}")
    print(f"Mejor valor encontrado: {problem.function(*w)}")
    print(f"Iteraciones: {len(optimizer.get_history())}")
    print(f"w óptimo: {problem.optimal_params}")

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
