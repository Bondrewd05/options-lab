"""Gera reports/greeks.png. Execute após instalar options-lab[plots]."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from options_lab import black_scholes_greeks


def create_figure():
    """Sensibilidades de uma opção comprada; muda apenas o preço do ativo."""
    spots = [60.0 + i * 0.25 for i in range(321)]
    parameters = dict(strike=100, time_to_maturity=1, rate=0.05, volatility=0.20)
    calls = [black_scholes_greeks(s, **parameters) for s in spots]
    puts = [black_scholes_greeks(s, **parameters, option_type="put") for s in spots]
    base = black_scholes_greeks(100, **parameters)
    blue, orange, purple = "#2563a6", "#b84c30", "#7041a6"
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.71, wspace=0.32)
        fig.suptitle("Como a opção reage ao preço do ativo", x=0.07, y=0.97, ha="left", fontsize=21, fontweight="bold")
        fig.text(0.07, 0.875, "Black–Scholes · strike 100 · prazo 1 ano · juros contínuos 5% · volatilidade 20%", color="#4b5563", fontsize=11)

        axes[0].plot(spots, [g.delta for g in calls], label="Call", color=blue, lw=2.5)
        axes[0].plot(spots, [g.delta for g in puts], label="Put", color=orange, lw=2.5)
        axes[0].scatter([100, 100], [base.delta, base.delta - 1], color=[blue, orange], zorder=4)
        axes[0].set(title="Delta", ylabel="Variação do preço por +1 u.m. no ativo", ylim=(-1.05, 1.05))
        axes[0].legend(frameon=False, loc="lower right")

        axes[1].plot(spots, [g.gamma for g in calls], color=purple, lw=2.5)
        axes[1].scatter([100], [base.gamma], color=purple, zorder=4)
        axes[1].set(title="Gamma · call e put", ylabel="Variação do delta por +1 u.m. no ativo", ylim=(0, None))

        axes[2].plot(spots, [g.vega * 0.01 for g in calls], color=purple, lw=2.5)
        axes[2].scatter([100], [base.vega * 0.01], color=purple, zorder=4)
        axes[2].set(title="Vega por 1 ponto percentual", ylabel="Variação do preço por +1 pp de vol. (u.m.)", ylim=(0, None))

        for ax in axes:
            ax.axvline(100, color="#64748b", linestyle="--", linewidth=1, alpha=0.6)
            ax.set(xlabel="Preço do ativo (u.m.)", xlim=(60, 140))
            ax.grid(axis="y", alpha=0.18)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        fig.text(0.07, 0.075, "Pontos: ativo a 100. As variações indicadas são aproximações locais, com as demais entradas fixas.", fontsize=10, color="#4b5563")
        fig.text(0.07, 0.025, "A API retorna vega por unidade decimal; o gráfico mostra vega × 0,01.  •  u.m. = unidade monetária", fontsize=10, color="#4b5563")
        return fig


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "reports" / "greeks.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Figura salva: {output}")
