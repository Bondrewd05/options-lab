"""Reproduz e desenha experimentos pareados de delta hedge sob Q."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from options_lab import run_hedging_experiments


BLUE = "#2563a6"
PURPLE = "#7041a6"
ORANGE = "#b84c30"
GREEN = "#217b6b"
GRAY = "#4b5563"
STYLE = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}


def _decimal(value, _position=None):
    return f"{value:g}".replace(".", ",")


def _figure(data, title, subtitle):
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7))
    figure.subplots_adjust(left=0.085, right=0.97, bottom=0.31, top=0.73, wspace=0.31)
    figure.suptitle(title, x=0.085, y=0.96, ha="left", fontsize=21, fontweight="bold")
    figure.text(0.085, 0.885, subtitle, fontsize=10, color=GRAY)
    count = f"{data['n_paths']:,}".replace(",", ".")
    figure.text(
        0.085, 0.835,
        f"{count} trajetórias sob Q · semente inicial {data['seed_start']} · "
        f"grade-mestra com {data['master_steps']} intervalos",
        fontsize=10, color=GRAY,
    )
    for axis in axes:
        axis.yaxis.set_major_formatter(FuncFormatter(_decimal))
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    return figure, axes


def _mean_errors(scenarios):
    means = [item["summary"]["mean_pnl"] for item in scenarios]
    intervals = [item["summary"]["mean_confidence_interval_95_approx"] for item in scenarios]
    errors = [
        [mean - interval[0] for mean, interval in zip(means, intervals)],
        [interval[1] - mean for mean, interval in zip(means, intervals)],
    ]
    return means, errors


def _frequency_axes(axes, frequencies):
    positions = list(range(len(frequencies)))
    for axis in axes:
        axis.set_xticks(positions, [str(value) for value in frequencies])
        axis.set_xlabel("Intervalos da grade N")
        axis.set_xlim(-0.2, len(frequencies) - 0.8)
    return positions


def create_frequency_figure(data):
    """Compara média, dispersão e RMSE do PnL terminal sem custos."""
    scenarios = sorted(
        (item for item in data["scenarios"]
         if item["family"] == "frequency_costs" and item["transaction_cost_rate"] == 0),
        key=lambda item: item["steps"],
    )
    with plt.rc_context(STYLE):
        figure, axes = _figure(
            data, "Frequência dos ajustes e resultado do hedge",
            "Sem custos · volatilidade do ativo e do hedge 20% · prêmio Black–Scholes com 20%",
        )
        positions = _frequency_axes(axes, [item["steps"] for item in scenarios])
        means, errors = _mean_errors(scenarios)
        mean_plot = axes[0].errorbar(
            positions, means, yerr=errors, fmt="o", color=BLUE,
            capsize=5, linewidth=1.7, label="Média e IC de 95% aproximado",
        )
        reference = axes[0].axhline(0, color=GRAY, linestyle="--", linewidth=1.3, label="Média teórica sob Q")
        axes[0].set(title="Média do PnL e incerteza amostral", ylabel="PnL no vencimento (u.m.)")
        deviation, = axes[1].plot(
            positions, [item["summary"]["std_pnl"] for item in scenarios],
            color=PURPLE, marker="o", linewidth=2, label="Desvio-padrão amostral",
        )
        rmse, = axes[1].plot(
            positions, [item["summary"]["rmse_pnl"] for item in scenarios],
            color=ORANGE, marker="s", fillstyle="none", linestyle="--",
            linewidth=1.6, label="RMSE em relação a zero",
        )
        axes[1].set(title="Dispersão e magnitude do erro", ylabel="PnL no vencimento (u.m.)", ylim=(0, None))
        figure.legend(handles=[mean_plot, reference, deviation, rmse],
                      loc="lower left", bbox_to_anchor=(0.075, 0.145), ncol=2, frameon=False)
        figure.text(0.085, 0.085, "Cada cenário usa subgrades das mesmas trajetórias. As grades 12, 52 e 252 não são aninhadas entre si.", fontsize=9, color=GRAY)
        figure.text(0.085, 0.04, "Vendedor de uma call · ativo e strike 100 · prazo 1 ano · juros 5% · u.m. = unidade monetária do vencimento.", fontsize=9, color=GRAY)
        return figure


def create_costs_figure(data):
    """Mostra RMSE e média terminal ao variar a frequência e o custo."""
    with plt.rc_context(STYLE):
        figure, axes = _figure(
            data, "Frequência dos ajustes com custos de negociação",
            "Volatilidade do ativo e do hedge 20% · prêmio Black–Scholes com 20% · delta Black–Scholes",
        )
        positions = _frequency_axes(axes, data["frequencies"])
        lines = []
        for cost, color, marker in zip(data["cost_rates"], [BLUE, GREEN, PURPLE, ORANGE], ["o", "s", "^", "D"]):
            scenarios = sorted(
                (item for item in data["scenarios"]
                 if item["family"] == "frequency_costs" and item["transaction_cost_rate"] == cost),
                key=lambda item: item["steps"],
            )
            line, = axes[0].plot(
                positions, [item["summary"]["rmse_pnl"] for item in scenarios],
                color=color, marker=marker, linewidth=1.8,
                label=f"κ = {cost * 10000:g} pb".replace(".", ","),
            )
            lines.append(line)
            means, errors = _mean_errors(scenarios)
            axes[1].errorbar(positions, means, yerr=errors, color=color,
                             marker=marker, capsize=3, linewidth=1.6)
        axes[0].set(title="RMSE em relação a zero", ylabel="PnL no vencimento (u.m.)", ylim=(0, None))
        axes[1].set(title="Média do PnL e IC de 95% aproximado", ylabel="PnL no vencimento (u.m.)")
        axes[1].axhline(0, color=GRAY, linestyle="--", linewidth=1)
        figure.legend(handles=lines, loc="lower left", bbox_to_anchor=(0.075, 0.17), ncol=4, frameon=False)
        figure.text(0.085, 0.10, "Custo proporcional ao valor negociado em ações, incluindo posição inicial, ajustes e liquidação final. 1 pb = 0,01%.", fontsize=9, color=GRAY)
        figure.text(0.085, 0.055, "Mesmas trajetórias em todos os cenários · custos pagos ao longo do tempo e refletidos no caixa do vencimento.", fontsize=9, color=GRAY)
        figure.text(0.085, 0.015, "Call · ativo e strike 100 · prazo 1 ano · juros 5% · u.m. = unidade monetária do vencimento.", fontsize=9, color=GRAY)
        return figure


def create_volatility_figure(data):
    """Separa o efeito do prêmio do efeito da volatilidade usada no hedge."""
    with plt.rc_context(STYLE):
        figure, axes = _figure(
            data, "Volatilidade do hedge e regra do prêmio recebido",
            "Volatilidade do ativo 20% · 252 intervalos · sem custos · duas regras para o prêmio inicial",
        )
        positions = list(range(len(data["hedge_volatilities"])))
        for axis in axes:
            axis.set_xticks(positions, [f"{value:.0%}" for value in data["hedge_volatilities"]])
            axis.set_xlabel("Volatilidade usada no hedge")
            axis.set_xlim(-0.3, len(positions) - 0.7)
        handles = []
        rules = [("fixed_true_model", "Prêmio fixo BS(20%)", BLUE, -0.045),
                 ("hedge_model", "Prêmio BS(σ do hedge)", PURPLE, 0.045)]
        for rule, label, color, offset in rules:
            scenarios = sorted(
                (item for item in data["scenarios"]
                 if item["family"] == "volatility" and item["premium_convention"] == rule),
                key=lambda item: item["hedge_volatility"],
            )
            means, errors = _mean_errors(scenarios)
            estimate = axes[0].errorbar(
                [position + offset for position in positions], means, yerr=errors,
                fmt="o", capsize=5, color=color, label=label,
            )
            theory, = axes[0].plot(
                positions, [item["theoretical_mean_without_costs"] for item in scenarios],
                color=color, linestyle="--", linewidth=1.3,
                label=f"Média teórica · {label.lower()}",
            )
            handles.extend([estimate, theory])
            if rule == "fixed_true_model":
                axes[1].plot(positions, [item["summary"]["std_pnl"] for item in scenarios],
                             color=GREEN, marker="o", linewidth=2)
        axes[0].set(title="Médias e IC de 95% aproximado", ylabel="PnL no vencimento (u.m.)")
        axes[1].set(title="Desvio-padrão nas duas regras de prêmio", ylabel="Desvio-padrão do PnL (u.m.)", ylim=(0, None))
        figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.145), ncol=2, frameon=False, fontsize=9)
        figure.text(0.085, 0.085, "Mesmas trajetórias e operações em cada par. Alterar apenas o prêmio desloca o PnL e preserva o desvio-padrão.", fontsize=9, color=GRAY)
        figure.text(0.085, 0.04, "Média teórica = diferença entre prêmio recebido e BS(20%), capitalizada até T. u.m. = moeda do vencimento.", fontsize=9, color=GRAY)
        return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", type=Path, help="Regenera figuras a partir de resultados já salvos.")
    parser.add_argument("--n-paths", type=int, default=2000, help="Número de trajetórias (padrão 2000).")
    parser.add_argument("--seed-start", type=int, default=2026, help="Semente inicial (padrão 2026).")
    arguments = parser.parse_args()
    output_dir = Path(__file__).resolve().parents[1] / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.from_json is None:
        data = run_hedging_experiments(arguments.n_paths, seed_start=arguments.seed_start)
        output_json = output_dir / "hedging_experiments_results.json"
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Resultados: {output_json}")
    else:
        data = json.loads(arguments.from_json.read_text(encoding="utf-8"))
    for filename, create in (
        ("hedging_frequency.png", create_frequency_figure),
        ("hedging_costs.png", create_costs_figure),
        ("hedging_volatility.png", create_volatility_figure),
    ):
        figure = create(data)
        output = output_dir / filename
        figure.savefig(output, dpi=160, facecolor="white")
        plt.close(figure)
        print(f"Figura: {output}")


if __name__ == "__main__":
    main()
