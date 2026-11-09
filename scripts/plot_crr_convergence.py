"""Compara a árvore CRR com Black–Scholes; execute com options-lab[plots]."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from options_lab import binomial_crr_price, black_scholes_price


def create_figure():
    """Preço e erro absoluto da call, preservando a oscilação par/ímpar."""
    parameters = dict(spot=100, strike=100, time_to_maturity=1, rate=0.05, volatility=0.20)
    reference = black_scholes_price(**parameters)
    steps = list(range(1, 101)) + [200, 201, 500, 501, 1000, 1001]
    prices = {n: binomial_crr_price(**parameters, steps=n) for n in steps}

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.23, top=0.70, wspace=0.27)
        figure.suptitle("Da árvore binomial a Black–Scholes", x=0.08, y=0.96, ha="left", fontsize=21, fontweight="bold")
        figure.text(0.08, 0.87, "Call europeia · ativo e strike 100 · prazo 1 ano · juros 5% · volatilidade 20%", fontsize=11, color="#4b5563")
        axes[0].axhline(reference, color="#111827", linestyle="--", linewidth=1.4, label=f"Black–Scholes: {reference:.4f}".replace(".", ","))
        for parity, label, color in [(0, "N par", "#2563a6"), (1, "N ímpar", "#b84c30")]:
            counts = [n for n in steps if n % 2 == parity]
            short_counts = [n for n in counts if n <= 100]
            axes[0].plot(short_counts, [prices[n] for n in short_counts], color=color, lw=1.8, label=label)
            axes[1].loglog(counts, [abs(prices[n] - reference) for n in counts], color=color, lw=1.8, label=label)
        axes[0].set(title="Preço calculado pela árvore", xlabel="Número de passos N", ylabel="Preço da call (u.m.)", xlim=(1, 100))
        axes[1].set(title="Distância até Black–Scholes", xlabel="Número de passos N (escala log)", ylabel="Erro absoluto (u.m., escala log)", xlim=(1, 1100))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=10)
        figure.text(0.08, 0.105, "Aumentar N encurta cada intervalo: Δt = 1/N ano. O prazo total continua sendo um ano.", fontsize=10, color="#4b5563")
        figure.text(0.08, 0.05, "A convergência pode oscilar: o erro não precisa diminuir a cada passo adicional. u.m. = unidade monetária.", fontsize=10, color="#4b5563")
        return figure


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "reports" / "crr_convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Figura salva: {output}")
