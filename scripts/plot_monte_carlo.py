"""Reproduz preços, incertezas e figura Monte Carlo; requer options-lab[plots]."""

import json
import math
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from options_lab import binomial_crr_price, black_scholes_price, monte_carlo_price


def run_experiment():
    """Mesmo seed em cada chamada: amostras maiores têm prefixos em comum."""
    parameters = dict(spot=100, strike=100, time_to_maturity=1, rate=0.05, volatility=0.20)
    seed = 2026
    counts = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
    calls = [asdict(monte_carlo_price(**parameters, n_paths=n, seed=seed)) for n in counts]
    put = asdict(monte_carlo_price(**parameters, n_paths=1_000_000, option_type="put", seed=seed))
    return {
        "schema_version": 1,
        "package_version": version("options-lab"),
        "python_version": platform.python_version(),
        "rng": "random.Random(seed).gauss(0, 1)",
        "parameters": parameters,
        "seed": seed,
        "sampling": "Amostras maiores reutilizam os sorteios iniciais; pontos não independentes.",
        "confidence_method": "Intervalo normal aproximado de 95%; média +/- z_0.975 * erro padrão.",
        "call_estimates": calls,
        "put_estimate": put,
        "references": {
            "black_scholes_call": black_scholes_price(**parameters),
            "black_scholes_put": black_scholes_price(**parameters, option_type="put"),
            "crr_steps": 1000,
            "crr_call": binomial_crr_price(**parameters, steps=1000),
            "crr_put": binomial_crr_price(**parameters, steps=1000, option_type="put"),
        },
    }


def create_figure(data):
    estimates = data["call_estimates"]
    counts = [estimate["n_paths"] for estimate in estimates]
    prices = [estimate["price"] for estimate in estimates]
    errors = [estimate["standard_error"] for estimate in estimates]
    lower = [estimate["confidence_interval"][0] for estimate in estimates]
    upper = [estimate["confidence_interval"][1] for estimate in estimates]
    reference = data["references"]["black_scholes_call"]
    blue, purple = "#2563a6", "#7041a6"
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 6))
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.25, top=0.71, wspace=0.29)
        figure.suptitle("Monte Carlo: preço e incerteza", x=0.08, y=0.96, ha="left", fontsize=22, fontweight="bold")
        figure.text(0.08, 0.87, "Call · ativo e strike 100 · prazo 1 ano · juros 5% · volatilidade 20% · semente 2026", fontsize=11, color="#4b5563")

        axes[0].fill_between(counts, lower, upper, color=blue, alpha=0.13, label="IC de 95% aproximado")
        axes[0].plot(counts, prices, color=blue, marker="o", lw=2, label="Preço estimado")
        axes[0].axhline(reference, color="#111827", ls="--", lw=1.4, label=f"Black–Scholes: {reference:.4f}".replace(".", ","))
        axes[0].set(xscale="log", title="Estimativa e intervalo de confiança", xlabel="Número de amostras M (escala log)", ylabel="Preço da call (u.m.)")
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        axes[1].loglog(counts, errors, color=purple, marker="o", lw=2, label="Erro padrão estimado")
        reference_errors = [errors[0] * math.sqrt(counts[0] / n) for n in counts]
        axes[1].loglog(counts, reference_errors, color="#475569", ls="--", lw=1.5, label="Referência proporcional a 1/√M")
        axes[1].set(title="Incerteza da média estimada", xlabel="Número de amostras M (escala log)", ylabel="Erro padrão (u.m., escala log)")
        axes[1].set_ylim(0.01, 1.0)
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.legend(frameon=False, fontsize=9, loc="best")
        figure.text(0.08, 0.13, "Mais amostras reduzem a incerteza típica; o erro observado contra Black–Scholes pode oscilar.", fontsize=10, color="#4b5563")
        figure.text(0.08, 0.08, "Cada amostra maior reutiliza os sorteios iniciais. Os pontos do gráfico são dependentes entre si.", fontsize=10, color="#4b5563")
        figure.text(0.08, 0.03, "O intervalo mede incerteza amostral dentro do modelo. u.m. = unidade monetária.", fontsize=10, color="#4b5563")
        return figure


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    data_path = output_dir / "monte_carlo_results.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure = create_figure(data)
    figure_path = output_dir / "monte_carlo_convergence.png"
    figure.savefig(figure_path, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Resultados: {data_path}")
    print(f"Figura: {figure_path}")
