"""Reproduz uma trajetória auditável e a distribuição do resultado de delta hedge."""

import json
import math
import platform
import statistics
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from options_lab import delta_hedge_path, simulate_gbm_path


def run_experiment():
    parameters = dict(spot=100.0, strike=100.0, time_to_maturity=1.0, rate=0.05, volatility=0.20)
    steps, n_paths, seed_start = 52, 2000, 2026
    pnls = []
    example = None
    for index in range(n_paths):
        path = simulate_gbm_path(
            parameters["spot"], parameters["time_to_maturity"], parameters["rate"],
            parameters["volatility"], steps=steps, seed=seed_start + index,
        )
        result = delta_hedge_path(
            path, parameters["strike"], parameters["time_to_maturity"],
            parameters["rate"], parameters["volatility"],
        )
        pnls.append(result.terminal_pnl)
        if index == 0:
            example = asdict(result)
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls)
    se = std / math.sqrt(n_paths)
    z = statistics.NormalDist().inv_cdf(0.975)
    percentiles = statistics.quantiles(pnls, n=100, method="inclusive")
    return {
        "schema_version": 1,
        "package_version": version("options-lab"),
        "python_version": platform.python_version(),
        "parameters": parameters,
        "measure": "Q: drift do ativo igual a rate",
        "steps": steps,
        "n_paths": n_paths,
        "seed_start": seed_start,
        "seed_recipe": "seed de cada trajetória = seed_start + índice, com índice entre 0 e n_paths-1",
        "conventions": {
            "position": "Vendedor de uma call europeia sobre uma unidade do ativo, sem dividendos.",
            "initial_premium": "Black–Scholes com a mesma volatilidade da simulação e do hedge.",
            "cash": "Juros contínuos à mesma taxa para aplicação e empréstimo; sem custos de negociação.",
            "record_timing": "Após juros e operações; no vencimento, após vender as ações e pagar a call.",
            "pnl_units": "Unidades monetárias do vencimento.",
        },
        "example": example,
        "terminal_pnls": pnls,
        "summary": {
            "mean_pnl": mean,
            "std_pnl": std,
            "rmse_pnl": math.sqrt(math.fsum(value * value for value in pnls) / n_paths),
            "mean_standard_error": se,
            "mean_confidence_interval_95_approx": [mean - z * se, mean + z * se],
            "percentile_5": percentiles[4],
            "median": statistics.median(pnls),
            "percentile_95": percentiles[94],
            "negative_fraction": sum(value < 0 for value in pnls) / n_paths,
            "minimum": min(pnls),
            "maximum": max(pnls),
        },
    }


def create_path_figure(data):
    records = data["example"]["records"]
    time = [row["time"] for row in records]
    colors = ["#2563a6", "#7041a6", "#b84c30", "#2563a6"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        figure.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.80, hspace=0.48, wspace=0.28)
        figure.suptitle("Acompanhando o vendedor de uma call", x=0.09, y=0.96, ha="left", fontsize=21, fontweight="bold")
        figure.text(0.09, 0.89, "Ativo e strike 100 · prazo 1 ano · juros 5% · volatilidade 20% · 52 intervalos · semente 2026", fontsize=10, color="#4b5563")
        specifications = [
            ("spot", "Preço do ativo", "Preço (u.m.)"),
            ("shares", "Posição após cada ajuste", "Quantidade de ações"),
            ("cash", "Saldo de caixa após operações", "Caixa (u.m.)"),
            ("net_value", "Valor líquido da estratégia", "Ações + caixa − obrigação (u.m.)"),
        ]
        for axis, (field, title, ylabel), color in zip(axes.flat, specifications, colors):
            values = [row[field] for row in records]
            if field == "shares":
                axis.step(time, values, where="post", color=color, linewidth=1.8)
            else:
                axis.plot(time, values, color=color, linewidth=1.8)
            axis.scatter([time[-1]], [values[-1]], s=28, color=color, zorder=4)
            axis.set(title=title, xlabel="Tempo decorrido (anos)", ylabel=ylabel, xlim=(0, 1))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
            axis.grid(alpha=0.18)
            if field == "spot":
                axis.axhline(data["parameters"]["strike"], color="#475569", linestyle="--", linewidth=1, label="Strike")
                axis.legend(frameon=False, fontsize=9)
            if field == "net_value":
                axis.axhline(0, color="#475569", linestyle="--", linewidth=1)
        figure.text(0.09, 0.075, "Registros nas datas da grade. A quantidade de ações fica constante entre ajustes; as demais linhas conectam os registros.", fontsize=9, color="#4b5563")
        figure.text(0.09, 0.03, "No último ponto, ações e obrigação já foram liquidadas. O caixa restante é o resultado final, em moeda do vencimento.", fontsize=9, color="#4b5563")
        return figure


def create_pnl_figure(data):
    pnls = data["terminal_pnls"]
    summary = data["summary"]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(10.5, 5.7))
        figure.subplots_adjust(left=0.09, right=0.97, bottom=0.23, top=0.73)
        figure.suptitle("O ajuste discreto deixa um resultado residual", x=0.09, y=0.96, ha="left", fontsize=20, fontweight="bold")
        figure.text(0.09, 0.87, "2.000 trajetórias sob Q · 52 intervalos por trajetória · mesmo modelo no ativo e no hedge", fontsize=10, color="#4b5563")
        _, bins, patches = axis.hist(pnls, bins=60, edgecolor="white", linewidth=0.5)
        for left, right, patch in zip(bins[:-1], bins[1:], patches):
            patch.set_facecolor("#b84c30" if (left + right) / 2 < 0 else "#2563a6")
        axis.axvline(0, color="#111827", linestyle="--", linewidth=1.4, label="Resultado zero")
        axis.axvline(summary["mean_pnl"], color="#7041a6", linewidth=1.6, label=f"Média: {summary['mean_pnl']:.4f}".replace(".", ","))
        axis.set(xlabel="Saldo final após vender ações e pagar a call (u.m.)", ylabel="Quantidade de trajetórias")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        axis.grid(axis="y", alpha=0.15)
        axis.legend(frameon=False, fontsize=10)
        std_label = f"{summary['std_pnl']:.4f}".replace(".", ",")
        se_label = f"{summary['mean_standard_error']:.4f}".replace(".", ",")
        figure.text(0.09, 0.105, f"Desvio-padrão dos resultados: {std_label} u.m. · Erro padrão da média: {se_label} u.m.", fontsize=10, color="#4b5563")
        figure.text(0.09, 0.05, "A média próxima de zero sob Q não implica resultado zero em cada trajetória. Não há custos de negociação neste cenário.", fontsize=9, color="#4b5563")
        return figure


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    output_json = output_dir / "delta_hedge_results.json"
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, create in (("delta_hedge_path.png", create_path_figure), ("delta_hedge_pnl.png", create_pnl_figure)):
        figure = create(data)
        output = output_dir / name
        figure.savefig(output, dpi=160, facecolor="white")
        plt.close(figure)
        print(f"Figura: {output}")
    print(f"Resultados: {output_json}")
    print(json.dumps(data["summary"], ensure_ascii=False))
