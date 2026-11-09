"""Árvore CRR didática de dois passos, com preços do ativo e da call."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from options_lab import binomial_crr_price, european_payoff


def create_figure():
    steps, maturity, spot, strike, rate, volatility = 2, 1.0, 100.0, 100.0, 0.05, 0.20
    dt = maturity / steps
    a = volatility * math.sqrt(dt)
    probability = math.expm1(rate * dt + a) / math.expm1(2 * a)
    figure, axis = plt.subplots(figsize=(11.5, 7))
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.18, top=0.76)
    figure.suptitle("Uma árvore de dois passos", x=0.06, y=0.95, ha="left", fontsize=22, fontweight="bold")
    figure.text(0.06, 0.87, "Strike 100 · prazo 1 ano · juros 5% · volatilidade 20% · cada passo dura 6 meses", fontsize=11, color="#4b5563")
    axis.set(xlim=(-0.45, 2.55), ylim=(-2.6, 2.9))
    axis.axis("off")

    for level in range(steps):
        for up_moves in range(level + 1):
            y = 2 * up_moves - level
            for increment in (-1, 1):
                axis.annotate("", xy=(level + 1, y + increment), xytext=(level, y),
                              arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.7, "shrinkA": 46, "shrinkB": 49})

    for level in range(steps + 1):
        for up_moves in range(level + 1):
            node_spot = spot * math.exp((2 * up_moves - level) * a)
            remaining = steps - level
            value = (
                binomial_crr_price(node_spot, strike, remaining * dt, rate, volatility, steps=remaining)
                if remaining else european_payoff(node_spot, strike)
            )
            label = f"S = {node_spot:.2f}\nC = {value:.4f}".replace(".", ",")
            color = "#eaf2fb" if remaining else "#fff0e5"
            axis.text(level, 2 * up_moves - level, label, ha="center", va="center", fontsize=12,
                      bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "#64748b", "lw": 1})
    for x, label in enumerate(("Hoje", "6 meses", "Vencimento · 1 ano")):
        axis.text(x, 2.65, label, ha="center", fontsize=12, fontweight="bold")
    axis.text(0.45, 0.87, f"p = {probability:.4f}".replace(".", ","), fontsize=10, color="#475569", ha="center")
    axis.text(0.45, -0.87, f"1 − p = {1 - probability:.4f}".replace(".", ","), fontsize=10, color="#475569", ha="center")
    figure.text(0.06, 0.115, "S: preço do ativo. C: preço da call; no vencimento, C é o payoff max(S − 100, 0).", fontsize=11, color="#334155")
    figure.text(0.06, 0.065, "Para obter C hoje, comece pelos pagamentos à direita e volte um passo de cada vez, descontando os valores.", fontsize=10, color="#4b5563")
    figure.text(0.06, 0.025, "p é o peso de alta usado na avaliação por ausência de arbitragem; cada nó tem os mesmos pesos.", fontsize=10, color="#4b5563")
    return figure


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "reports" / "crr_tree.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Figura salva: {output}")
