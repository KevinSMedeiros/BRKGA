"""Gera gráficos pareados de BRKGA e guloso a partir dos CSVs do comparador."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
TIME = {"A": "Tempo A(ms)", "B": "Tempo B(ms)"}
ACTIVE_ARCS = ["Arcos>0 A", "Arcos>0 B"]
WAYPOINTS = ["NumWayPoints A", "NumWayPoints B"]
ARC_DELTA = "Diferença de arcos (A−B)"
WP_DELTA = "Diferença de waypoints (B−A)"
SPEEDUP = "Speedup A/B"
RANK = "1a dif."
GAIN = "Ganho (%)"
SIGNED_GAIN = "Ganho assinado (%)"
NUMERIC_COLUMNS = ["Arcs", *WAYPOINTS, *TIME.values(), *ACTIVE_ARCS, RANK]
REQUIRED_COLUMNS = {"Instancia", "Demanda", "Ganho", "Melhor", *NUMERIC_COLUMNS}

ALGORITHMS = {"A": "BRKGA", "B": "Guloso"}
WINNER_LABELS = {"A": "Vitória do BRKGA", "B": "Vitória do guloso"}
WINNER_COLORS = {"A": "#226a9f", "B": "#b6531c"}
SET_COLORS = {"setA": "#226a9f", "setB": "#b6531c", "setC": "#29805e", "setX": "#7957a8"}
NEUTRAL = "#657486"
OUTCOME_COLORS = {"Empate": NEUTRAL, "Com vencedor": "#267b6b"}
GRID = "#dce3eb"
INK = "#1c2c3e"
STYLE = {
    "axes.axisbelow": True,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "medium",
    "font.size": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
}
Manifest = list[dict[str, object]]


def parse_number(value: object) -> float:
    """Extrai números do CSV, inclusive ganhos em notação científica."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?", str(value))
    if match is None:
        return np.nan
    return float(match.group(0).replace(",", "."))


def read_comparisons(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Lê uma linha por demanda pareada, conforme exportada pelo comparador."""
    files = sorted(input_dir.glob("set*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo set*.csv encontrado em {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} não contém as colunas {', '.join(sorted(missing))}")
        frame["Conjunto"] = path.stem
        frame["Fonte"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        data[column] = data[column].map(parse_number)

    data[GAIN] = data["Ganho"].map(parse_number).abs()
    data[SIGNED_GAIN] = data[GAIN].where(data["Melhor"].eq("A"), -data[GAIN])
    data[ARC_DELTA] = data[ACTIVE_ARCS[0]] - data[ACTIVE_ARCS[1]]
    data[WP_DELTA] = data[WAYPOINTS[1]] - data[WAYPOINTS[0]]
    times = data[list(TIME.values())]
    valid_times = times.gt(0).all(axis=1)
    data[SPEEDUP] = times.loc[valid_times, TIME["A"]] / times.loc[valid_times, TIME["B"]]
    return data, [path.name for path in files]


def save_chart(
    figure: plt.Figure, output_dir: Path, filename: str, title: str,
    data: pd.DataFrame, columns: str, scope: str, manifest: Manifest,
) -> None:
    sources = ", ".join(sorted(data["Fonte"].unique()))
    source_label = f"Fonte  {sources}  ·  " if sources else ""
    figure.suptitle(title, x=0.025, y=0.987, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.025, 0.018, f"{source_label}{len(data)} demandas\n{scope}",
        fontsize=8, color=NEUTRAL, ha="left", va="bottom",
    )
    figure.tight_layout(rect=(0.01, 0.081, 0.99, 0.94))
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    manifest.append(
        {"arquivo": filename, "titulo": title, "fontes": sources,
         "campos": columns, "recorte": scope, "registros": len(data)}
    )


def set_panels(
    data: pd.DataFrame, *, sharex: bool = False, sharey: bool = False, panel_width: float = 5.7,
):
    sets = sorted(data["Conjunto"].unique())
    columns = min(2, max(1, len(sets)))
    rows = max(1, (len(sets) + columns - 1) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(panel_width * columns, 3.8 * rows),
        sharex=sharex, sharey=sharey, squeeze=False,
    )
    for axis in axes.flat[len(sets):]:
        axis.set_visible(False)
    return figure, zip(sets, axes.flat)


def winner_legend(axis: plt.Axes) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=WINNER_COLORS[winner], markeredgecolor="none", label=label)
        for winner, label in WINNER_LABELS.items()
    ]
    axis.legend(handles=handles)


def decided_with_gain(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data["Melhor"].isin(ALGORITHMS) & data[GAIN].notna()]


def offsets_for_points(size: int, width: float) -> np.ndarray:
    return np.linspace(-width, width, size) if size > 1 else np.zeros(size)


def plot_active_arcs(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    decided = data.loc[data["Melhor"].isin(ALGORITHMS)].dropna(subset=ACTIVE_ARCS)
    figure, panels = set_panels(data, sharex=True, sharey=True, panel_width=4.35)
    limit = max(1, decided[ACTIVE_ARCS].max().max()) * 1.07
    for set_name, axis in panels:
        subset = decided.loc[decided["Conjunto"].eq(set_name)]
        frequencies = subset.groupby([*ACTIVE_ARCS, "Melhor"]).size().rename("n").reset_index()
        mixed = frequencies.groupby(ACTIVE_ARCS)["Melhor"].transform("size").gt(1)
        axis.plot([0, limit], [0, limit], "--", color=NEUTRAL, lw=1)
        for winner, points in frequencies.loc[~mixed].groupby("Melhor"):
            axis.scatter(
                points[ACTIVE_ARCS[0]], points[ACTIVE_ARCS[1]],
                s=35 * points["n"], color=WINNER_COLORS[winner], alpha=0.8,
                edgecolors="white", linewidths=0.5,
            )
        for (arcs_a, arcs_b), count in frequencies.loc[mixed].groupby(ACTIVE_ARCS)["n"].sum().items():
            axis.plot(
                arcs_a, arcs_b, marker="o", fillstyle="left", markersize=np.sqrt(35 * count),
                markerfacecolor=WINNER_COLORS["A"], markerfacecoloralt=WINNER_COLORS["B"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=3,
            )
        axis.set(
            title=f"{set_name}  ·  n={len(subset)}", xlim=(0, limit), ylim=(0, limit),
            xlabel="Arcos usados pelo BRKGA", ylabel="Arcos usados pelo guloso", aspect="equal",
        )
        axis.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(6, integer=True))
        axis.grid()
    winner_legend(figure.axes[0])
    save_chart(
        figure, output_dir, "01_arcos_usados_pareados.png", "Arcos usados por demanda",
        decided, "Arcos>0 A, Arcos>0 B, Melhor",
        "Acima da diagonal, o guloso usa mais arcos. O tamanho indica a frequência. "
        "Pontos bicolores incluem vitórias de ambos.", manifest,
    )


def plot_runtime_by_edges(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid_arcs = data.loc[data["Arcs"].gt(0)]
    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    for winner, time in TIME.items():
        subset = valid_arcs.loc[valid_arcs[time].gt(0)]
        color = WINNER_COLORS[winner]
        axis.scatter(subset["Arcs"], subset[time], s=19, color=color, alpha=0.33)
        medians = subset.groupby(["Instancia", "Arcs"], as_index=False)[time].median()
        axis.scatter(
            medians["Arcs"], medians[time], s=43, marker="D", color=color,
            edgecolors="white", linewidths=0.65, label=ALGORITHMS[winner], zorder=3,
        )
    axis.set(
        xscale="log", yscale="log", xlabel="Número de arcos da instância (escala log)",
        ylabel="Tempo de execução (ms, escala log)",
    )
    axis.grid()
    axis.legend(loc="upper left")
    valid = valid_arcs.loc[valid_arcs[list(TIME.values())].gt(0).any(axis=1)]
    save_chart(
        figure, output_dir, "04_tempo_vs_arestas.png", "Tempo de execução e tamanho da instância",
        valid, "Instancia, Arcs, Tempo A(ms), Tempo B(ms)",
        "Círculos representam demandas. Losangos representam medianas por instância e algoritmo.",
        manifest,
    )


def plot_paired_runtime(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    time_a, time_b = TIME.values()
    valid = data.loc[data[[time_a, time_b]].gt(0).all(axis=1)]
    figure, axis = plt.subplots(figsize=(7.5, 6.1))
    for set_name, subset in valid.groupby("Conjunto"):
        axis.scatter(
            subset[time_a], subset[time_b], color=SET_COLORS.get(set_name, NEUTRAL),
            s=24, alpha=0.67, label=set_name,
        )
    if not valid.empty:
        low = valid[[time_a, time_b]].min().min()
        high = valid[[time_a, time_b]].max().max()
        axis.plot([low, high], [low, high], "--", color=NEUTRAL, label="Tempos iguais")
        axis.legend(loc="upper left")
    axis.set(
        xscale="log", yscale="log", xlabel="Tempo do BRKGA (ms, escala log)",
        ylabel="Tempo do guloso (ms, escala log)",
    )
    axis.grid()
    save_chart(
        figure, output_dir, "05_tempo_A_vs_B.png", "Comparação pareada dos tempos de execução",
        valid, "Tempo A(ms), Tempo B(ms)",
        "Demandas com ambos os tempos positivos. Abaixo da diagonal, o guloso foi mais rápido.",
        manifest,
    )


def plot_speedup_by_edges(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid = data.loc[data["Arcs"].gt(0) & data[SPEEDUP].gt(0)]
    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    for set_name, subset in valid.groupby("Conjunto"):
        axis.scatter(
            subset["Arcs"], subset[SPEEDUP], color=SET_COLORS.get(set_name, NEUTRAL),
            s=26, alpha=0.69, label=set_name,
        )
    axis.axhline(1, color=NEUTRAL, ls="--", lw=1)
    axis.set(
        xscale="log", yscale="log", xlabel="Número de arcos da instância (escala log)",
        ylabel="Tempo do BRKGA / tempo do guloso (escala log)",
    )
    axis.grid()
    if not valid.empty:
        axis.legend()
    save_chart(
        figure, output_dir, "06_speedup_vs_arestas.png", "Speedup do guloso e tamanho da instância",
        valid, "Arcs, Tempo A(ms), Tempo B(ms)",
        "Cada ponto representa uma demanda. Acima de 1×, o guloso foi mais rápido.", manifest,
    )


def plot_signed_gain(
    data: pd.DataFrame, output_dir: Path, manifest: Manifest,
    *, x: str, xlabel: str, filename: str, title: str, fields: str,
) -> None:
    decided = decided_with_gain(data).dropna(subset=[x])
    if x == "Arcs":
        decided = decided.loc[decided[x].gt(0)]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset[x], subset[SIGNED_GAIN], color=WINNER_COLORS[winner], s=28,
            alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3,
        )
    axis.axhline(0, color=NEUTRAL, ls="--", lw=1)
    if x == "Arcs":
        axis.set_xscale("log")
    else:
        axis.axvline(0, color=NEUTRAL, ls=":", lw=1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set(xlabel=xlabel, ylabel="Ganho assinado na primeira diferença (%)")
    axis.grid()
    winner_legend(axis)
    save_chart(
        figure, output_dir, filename, title, decided, fields,
        "Demandas decididas. Ganho positivo indica vitória do BRKGA. "
        "Ganho negativo indica vitória do guloso.", manifest,
    )


def plot_gain_by_winner(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    decided = decided_with_gain(data)
    figure, panels = set_panels(data, sharex=True)
    for set_name, axis in panels:
        subset = decided.loc[decided["Conjunto"].eq(set_name)]
        labels = []
        for row, winner in enumerate(reversed(ALGORITHMS)):
            values = subset.loc[subset["Melhor"].eq(winner), GAIN]
            labels.append(f"{ALGORITHMS[winner]}  (n={len(values)})")
            color = WINNER_COLORS[winner]
            # Separe pontos próximos na vertical preservando o ganho de cada vitória.
            for _, close_values in values.groupby(values.floordiv(2.5)):
                axis.scatter(
                    close_values, row + offsets_for_points(len(close_values), 0.28),
                    s=24, color=color, alpha=0.8, edgecolors="white", linewidths=0.4,
                )
            if not values.empty:
                axis.scatter(values.median(), row + 0.39, marker="D", s=36, color=color)
            else:
                axis.text(3, row, "Sem vitórias", va="center", color=NEUTRAL, fontsize=9)
        axis.set(
            title=set_name, yticks=[0, 1], yticklabels=labels, ylim=(-0.45, 1.56),
            xlim=(-2, 102), xticks=[0, 20, 40, 60, 80, 100],
            xlabel="Ganho na primeira diferença (%)",
        )
        axis.tick_params(labelbottom=True)
        axis.grid(axis="x")
    save_chart(
        figure, output_dir, "08_distribuicao_ganho_por_vencedor.png",
        "Distribuição do ganho nas vitórias de cada algoritmo", decided, "Conjunto, Ganho, Melhor",
        "Cada círculo representa uma vitória. Os losangos indicam medianas por vencedor e conjunto.",
        manifest,
    )


def first_difference_panels(data: pd.DataFrame):
    decided = decided_with_gain(data).dropna(subset=[RANK])
    ranks = sorted(decided[RANK].astype(int).unique())
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    for (winner, algorithm), axis in zip(ALGORITHMS.items(), axes):
        sample_size = decided["Melhor"].eq(winner).sum()
        axis.set(
            title=f"{algorithm}  ·  {sample_size} vitórias", ylabel="Ganho (%)",
            ylim=(-4, 115), yticks=[0, 25, 50, 75, 100], xticks=ranks,
            xlim=(-0.9, max(ranks, default=1) + 0.9),
        )
        axis.tick_params(labelbottom=True)
        axis.grid()
    axes[-1].set_xlabel("Posição da primeira diferença (índice iniciado em zero)")
    return decided, figure, axes


def plot_gain_by_first_difference(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    decided, figure, axes = first_difference_panels(data)
    scope = (
        "Posições mais à esquerda têm prioridade lexicográfica. "
        "A área do ponto indica quantas vitórias coincidem."
    )
    smallest_gain = decided.loc[decided[GAIN].between(0, 1e-10, inclusive="neither")].nsmallest(1, GAIN)
    for winner, axis in zip(ALGORITHMS, axes):
        subset = decided.loc[decided["Melhor"].eq(winner)]
        points = subset.groupby([RANK, GAIN]).size().rename("n").reset_index()
        axis.scatter(
            points[RANK], points[GAIN], s=34 * points["n"], color=WINNER_COLORS[winner],
            alpha=0.83, edgecolors="white", linewidths=0.55, zorder=3,
        )
        highlight = smallest_gain.loc[smallest_gain["Melhor"].eq(winner)]
        if not highlight.empty:
            axis.scatter(
                highlight[RANK], highlight[GAIN], s=100, facecolors="none",
                edgecolors=INK, linewidths=1, zorder=4,
            )
            row = highlight.iloc[0]
            amount = f"{row[GAIN]:.1e}".replace(".", ",").replace("-", "−")
            demand = row["Demanda"].replace("->", "→")
            scope += f"\nCírculo vazado indica {amount}% em {row['Instancia']} · {demand} (posição {int(row[RANK])})."
    save_chart(
        figure, output_dir, "09_ganho_vs_primeira_diferenca.png",
        "Ganho e posição da primeira diferença", decided, "1a dif., Ganho, Melhor", scope, manifest,
    )


def plot_first_difference_counts(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    decided = data.loc[data["Melhor"].isin(ALGORITHMS)].dropna(subset=[RANK])
    counts = pd.crosstab(decided[RANK].astype(int), decided["Melhor"]).reindex(
        columns=list(ALGORITHMS), fill_value=0,
    )
    figure, axis = plt.subplots(figsize=(9, 5.2))
    positions = np.arange(len(counts))
    axis.bar(positions, counts["A"], color=WINNER_COLORS["A"], label=ALGORITHMS["A"])
    axis.bar(
        positions, counts["B"], bottom=counts["A"], color=WINNER_COLORS["B"],
        label=ALGORITHMS["B"],
    )
    axis.set(
        xticks=positions, xticklabels=counts.index,
        xlabel="Posição da primeira diferença (índice iniciado em zero)", ylabel="Número de demandas",
    )
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y")
    axis.legend()
    save_chart(
        figure, output_dir, "10_primeira_diferenca_por_vencedor.png",
        "Frequência da primeira diferença por posição e vencedor", decided, "1a dif., Melhor",
        "Demandas decididas com primeira diferença registrada.", manifest,
    )


def stacked_counts(counts: pd.DataFrame, colors: list[str], xlabel: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.6, 2.5 + 0.55 * len(counts)))
    total = counts.sum(axis=1)
    shares = counts.div(total.replace(0, np.nan), axis=0).fillna(0).mul(100)
    y = np.arange(len(counts))
    left = np.zeros(len(counts))
    for name, color in zip(counts, colors):
        widths = shares[name].to_numpy()
        axis.barh(y, widths, left=left, height=0.62, color=color, label=name)
        for index, (start, width, count) in enumerate(zip(left, widths, counts[name])):
            if width >= 1.45 * len(str(count)):
                axis.text(
                    start + width / 2, index, str(count), color="white", ha="center",
                    va="center", fontsize=9, fontweight="bold",
                )
        left += widths
    for index, size in enumerate(total):
        axis.text(102, index, f"n={size}", color=NEUTRAL, va="center", fontsize=9)
    axis.set(
        xlim=(0, 110), xticks=np.arange(0, 101, 20), yticks=y, yticklabels=counts.index,
        xlabel=xlabel,
    )
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.grid(axis="x")
    axis.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
        ncol=2 if len(counts.columns) > 3 else len(counts.columns), fontsize=9,
    )
    return figure


def plot_waypoints_paired(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid = data.dropna(subset=WAYPOINTS)
    categories = ["Mais waypoints no BRKGA", "Mesma quantidade", "Mais waypoints no guloso"]
    comparison = pd.Series(
        np.select([valid[WP_DELTA].lt(0), valid[WP_DELTA].gt(0)], [categories[0], categories[2]],
                  default=categories[1]),
        index=valid.index,
    )
    counts = pd.crosstab(valid["Conjunto"], comparison).reindex(columns=categories, fill_value=0)
    counts.loc["Total"] = counts.sum(axis=0)
    figure = stacked_counts(
        counts, [WINNER_COLORS["A"], NEUTRAL, WINNER_COLORS["B"]], "Proporção das demandas",
    )
    save_chart(
        figure, output_dir, "11_waypoints_A_vs_B.png", "Qual algoritmo usa mais waypoints?",
        valid, "Conjunto, NumWayPoints A, NumWayPoints B",
        "Comparação pareada. Os números nas barras indicam contagens de demandas.", manifest,
    )


def plot_waypoint_difference_vs_speedup(
    data: pd.DataFrame, output_dir: Path, manifest: Manifest,
) -> None:
    valid = data.loc[data[SPEEDUP].gt(0)].dropna(subset=[WP_DELTA])
    xlimits = (min(0, valid[WP_DELTA].min()) - 0.45, max(0, valid[WP_DELTA].max()) + 0.45)
    ylimits = (min(1, valid[SPEEDUP].min()) / 1.25, max(1, valid[SPEEDUP].max()) * 1.25)
    ticks = sorted({0, *valid[WP_DELTA].astype(int)})
    xlabel = "Diferença de waypoints (guloso − BRKGA)"
    ylabel = "Tempo do BRKGA / tempo do guloso (escala log)"
    columns = "NumWayPoints A, NumWayPoints B, Tempo A(ms), Tempo B(ms)"

    figure, axis = plt.subplots(figsize=(9, 6))
    for set_name, subset in valid.groupby("Conjunto"):
        axis.scatter(
            subset[WP_DELTA], subset[SPEEDUP], color=SET_COLORS.get(set_name, NEUTRAL),
            s=26, alpha=0.69, label=set_name,
        )
    axis.axhline(1, color=NEUTRAL, ls="--", lw=1)
    axis.axvline(0, color=NEUTRAL, ls=":", lw=1)
    axis.set(
        xlim=xlimits, ylim=ylimits, xticks=ticks, yscale="log", xlabel=xlabel, ylabel=ylabel,
    )
    axis.grid()
    if not valid.empty:
        axis.legend(loc="upper right")
    save_chart(
        figure, output_dir, "12_diferenca_waypoints_vs_speedup.png",
        "Diferença de waypoints e speedup", valid, columns,
        "Cada ponto representa uma demanda. Acima de 1×, o guloso foi mais rápido.", manifest,
    )

    for set_name in sorted(data["Conjunto"].unique()):
        subset = valid.loc[valid["Conjunto"].eq(set_name)]
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        color = SET_COLORS.get(set_name, NEUTRAL)
        for difference, group in subset.groupby(WP_DELTA):
            axis.scatter(
                difference + offsets_for_points(len(group), 0.18), group[SPEEDUP],
                color=color, s=24, alpha=0.63,
            )
        medians = subset.groupby(WP_DELTA)[SPEEDUP].median()
        axis.scatter(
            medians.index, medians, marker="D", s=44, color=INK,
            edgecolors="white", linewidths=0.6, label="Mediana", zorder=3,
        )
        axis.axhline(1, color=NEUTRAL, ls="--", lw=1)
        axis.axvline(0, color=NEUTRAL, ls=":", lw=1)
        axis.set(
            xlim=xlimits, ylim=ylimits, xticks=ticks, yscale="log",
            xlabel=xlabel, ylabel=ylabel,
        )
        axis.grid()
        if not subset.empty:
            axis.legend()
        save_chart(
            figure, output_dir, f"12_diferenca_waypoints_vs_speedup_{set_name}.png",
            f"Diferença de waypoints e speedup  ·  {set_name}",
            subset, columns,
            "Cada círculo representa uma demanda. Pequenos deslocamentos horizontais revelam sobreposições.",
            manifest,
        )


def plot_outcomes_by_set(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    outcomes = {**WINNER_LABELS, "Empate": "Empate"}
    valid = data.loc[data["Melhor"].isin(outcomes)]
    counts = pd.crosstab(valid["Conjunto"], valid["Melhor"]).reindex(
        columns=list(outcomes), fill_value=0,
    ).rename(columns=outcomes)
    figure = stacked_counts(
        counts, [WINNER_COLORS["A"], WINNER_COLORS["B"], NEUTRAL], "Proporção das demandas",
    )
    save_chart(
        figure, output_dir, "14_resultados_por_conjunto.png",
        "Vitórias e empates por conjunto", valid, "Conjunto, Melhor",
        "Os números nas barras indicam contagens de demandas.", manifest,
    )


def plot_waypoints_vs_active_arcs(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid = data.dropna(subset=[WP_DELTA, ARC_DELTA])
    counts = valid.groupby([WP_DELTA, ARC_DELTA]).size().rename("n").reset_index()
    figure, axis = plt.subplots(figsize=(8.8, 5.9))
    axis.axhline(0, color=NEUTRAL, ls="--", lw=1)
    axis.axvline(0, color=NEUTRAL, ls="--", lw=1)
    axis.scatter(
        counts[WP_DELTA], -counts[ARC_DELTA], s=24 * np.sqrt(counts["n"]),
        color="#327b76", edgecolors="white", linewidths=0.65, alpha=0.84, zorder=3,
    )
    zero_count = counts.loc[counts[WP_DELTA].eq(0) & counts[ARC_DELTA].eq(0), "n"]
    if not zero_count.empty:
        axis.annotate(
            f"{zero_count.iloc[0]} demandas", xy=(0, 0), xytext=(14, 15),
            textcoords="offset points", fontsize=9,
            arrowprops={"arrowstyle": "-", "color": NEUTRAL},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
    axis.set(
        xlabel="Diferença de waypoints (guloso − BRKGA)",
        ylabel="Diferença de arcos usados (guloso − BRKGA)",
    )
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid()
    save_chart(
        figure, output_dir, "17_diferenca_waypoints_vs_arcos_usados.png",
        "Diferença de waypoints e diferença de arcos usados", valid,
        "NumWayPoints A, NumWayPoints B, Arcos>0 A, Arcos>0 B",
        "Valores positivos indicam mais waypoints ou arcos no guloso. "
        "A área do ponto cresce com a raiz da contagem.", manifest,
    )


def plot_tie_profiles(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    ties = data.loc[data["Melhor"].eq("Empate")]
    route_a, route_b = "Rota A", "Rota B"
    routes = ties.reindex(columns=[route_a, route_b])
    known = routes.notna().all(axis=1)
    same = routes[route_a].eq(routes[route_b])
    direct = ties[WAYPOINTS].eq(0).all(axis=1)
    categories = [
        "Mesma rota, zero waypoints",
        "Mesma rota, com waypoints",
        "Rotas de segmentos diferentes",
        "Rota indisponível",
    ]
    profile = pd.Series(
        np.select([known & same & direct, known & same, known], categories[:3], default=categories[3]),
        index=ties.index,
    )
    counts = pd.crosstab(ties["Conjunto"], profile).reindex(
        index=sorted(data["Conjunto"].unique()), columns=categories, fill_value=0,
    )
    colors = ["#2b668c", "#267b6b", "#8055a1", NEUTRAL]
    if counts[categories[-1]].sum() == 0:
        counts = counts.drop(columns=categories[-1])
        colors.pop()
    figure = stacked_counts(counts, colors, "Proporção dos empates no conjunto")
    save_chart(
        figure, output_dir, "18_perfil_empates.png", "Rotas de segmentos nos empates",
        ties, "Conjunto, Melhor, Rota A, Rota B, NumWayPoints A, NumWayPoints B",
        "Empate pelo vetor de utilização. Os números nas barras indicam contagens de empates.",
        manifest,
    )


def plot_gain_boxplot(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    decided, figure, axes = first_difference_panels(data)
    for winner, axis in zip(ALGORITHMS, axes):
        subset = decided.loc[decided["Melhor"].eq(winner)]
        color = WINNER_COLORS[winner]
        groups = subset.groupby(RANK)[GAIN]
        boxes = [(rank, gains) for rank, gains in groups if len(gains) >= 2]
        if boxes:
            axis.boxplot(
                [gains for _, gains in boxes], positions=[rank for rank, _ in boxes],
                widths=0.63, manage_ticks=False, patch_artist=True, showfliers=False,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.18},
                medianprops={"color": color, "linewidth": 1.6},
                whiskerprops={"color": color}, capprops={"color": color},
            )
        for rank, gains in groups:
            axis.scatter(
                rank + offsets_for_points(len(gains), 0.22), gains, s=19, color=color,
                alpha=0.76, edgecolors="white", linewidths=0.45, zorder=3,
            )
            axis.text(rank, 107, f"n={len(gains)}", ha="center", va="center", fontsize=7.5, color=NEUTRAL)
    save_chart(
        figure, output_dir, "19_boxplot_ganho_por_primeira_diferenca.png",
        "Distribuição do ganho por posição da primeira diferença", decided, "1a dif., Ganho, Melhor",
        "Caixas mostram mediana e quartis para n ≥ 2. Bigodes de até 1,5 × IQR. "
        "Todos os ganhos aparecem como pontos.", manifest,
    )


def plot_waypoints_by_outcome(
    data: pd.DataFrame, output_dir: Path, manifest: Manifest, *, split_wins: bool = False,
) -> None:
    valid = data.loc[data["Melhor"].isin([*ALGORITHMS, "Empate"])].dropna(subset=WAYPOINTS)
    observed = valid[WAYPOINTS].to_numpy().ravel()
    positions = np.arange(int(observed.max()) + 1) if observed.size else np.array([0])

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), sharey=True)
    colors = {"Empate": NEUTRAL, "Venceu": "#267b6b", "Perdeu": "#a84727"} if split_wins else OUTCOME_COLORS
    width = 0.72 / len(colors)
    offsets = (np.arange(len(colors)) - (len(colors) - 1) / 2) * width
    for axis, (side, algorithm), column in zip(axes, ALGORITHMS.items(), WAYPOINTS):
        if split_wins:
            labels = {key: "Venceu" if key == side else "Perdeu" for key in ALGORITHMS}
            outcomes = valid["Melhor"].map({**labels, "Empate": "Empate"})
        else:
            outcomes = valid["Melhor"].eq("Empate").map({True: "Empate", False: "Com vencedor"})
        totals = outcomes.value_counts()
        counts = pd.crosstab(valid[column].astype(int), outcomes).reindex(positions, fill_value=0)
        for offset, (outcome, color) in zip(offsets, colors.items()):
            group_size = totals.get(outcome, 0)
            frequency = counts.reindex(columns=[outcome], fill_value=0)[outcome]
            proportions = frequency / group_size * 100 if group_size else frequency
            bars = axis.bar(
                positions + offset, proportions, width=width, color=color,
                label=f"{outcome} (n={group_size})",
            )
            axis.bar_label(bars, labels=[str(n) if n else "" for n in frequency], padding=3, fontsize=8)
        axis.set(
            title=algorithm, xlabel="Número de waypoints", xticks=positions,
            ylim=(0, 108), yticks=np.arange(0, 101, 20),
        )
        axis.grid(axis="y")
        if split_wins:
            axis.legend(loc="upper right")
    axes[0].set_ylabel("Proporção das demandas dentro do grupo")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if split_wins:
        filename = "23_waypoints_empates_vitorias_derrotas.png"
        title = "Waypoints nos empates, vitórias e derrotas"
    else:
        axes[1].legend(loc="upper right")
        filename = "20_waypoints_empates_vs_decididas.png"
        title = "Waypoints nas demandas empatadas e com vencedor"
    save_chart(
        figure, output_dir, filename, title, valid,
        "NumWayPoints A, NumWayPoints B, Melhor",
        "Cada grupo soma 100% por algoritmo. Os números sobre as barras indicam contagens de demandas.",
        manifest,
    )


def plot_paired_waypoint_heatmap(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid = data.loc[data["Melhor"].isin([*ALGORITHMS, "Empate"])].dropna(subset=WAYPOINTS)
    grouped = valid.assign(Empate=valid["Melhor"].eq("Empate")).groupby(WAYPOINTS)["Empate"]
    stats = grouped.agg(total="size", taxa="mean")
    x, y = (np.arange(int(maximum) + 1) for maximum in valid[WAYPOINTS].max().fillna(0))
    rates = stats["taxa"].unstack(WAYPOINTS[0]).reindex(index=y, columns=x)

    figure, axis = plt.subplots(figsize=(8.4, 6))
    colors = matplotlib.colormaps["Blues"].with_extremes(bad="#e7ebef")
    heatmap = axis.imshow(
        np.ma.masked_invalid(rates.to_numpy()), origin="lower", cmap=colors,
        vmin=0, vmax=1, interpolation="nearest", aspect="auto",
    )
    axis.set(
        xlabel="Waypoints do BRKGA", ylabel="Waypoints do guloso", xticks=x, yticks=y,
    )
    axis.set_xticks(np.arange(len(x) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(y) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    for (brkga, guloso), row in stats.iterrows():
        rate = f"{row.taxa * 100:.1f}".removesuffix(".0").replace(".", ",")
        axis.text(
            brkga, guloso, f"{rate}%\nn={int(row.total)}", ha="center", va="center",
            color="white" if row.taxa >= 0.65 else INK, fontsize=9,
        )
    figure.colorbar(
        heatmap, ax=axis, label="Proporção de empates",
        format=PercentFormatter(xmax=1, decimals=0),
    )
    save_chart(
        figure, output_dir, "21_mapa_calor_waypoints_pareados.png",
        "Proporção de empates por par de waypoints", valid, "NumWayPoints A, NumWayPoints B, Melhor",
        "Cada célula mostra a proporção de empates e o total de demandas com aquele par de contagens.\n"
        "Células cinzas indicam pares sem observações.", manifest,
    )


def plot_active_arc_fraction_by_outcome(data: pd.DataFrame, output_dir: Path, manifest: Manifest) -> None:
    valid = data.loc[
        data["Melhor"].isin([*ALGORITHMS, "Empate"]) & data["Arcs"].gt(0)
        & data[ACTIVE_ARCS].ge(0).all(axis=1)
    ]
    fractions = valid[ACTIVE_ARCS].div(valid["Arcs"], axis=0).mul(100)
    outcomes = valid["Melhor"].eq("Empate").map({True: "Empate", False: "Com vencedor"})
    use_log = not valid.empty and fractions.gt(0).all().all()

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), sharex=True, sharey=True)
    for axis, (algorithm, column) in zip(axes, zip(ALGORITHMS.values(), ACTIVE_ARCS)):
        for outcome, color in OUTCOME_COLORS.items():
            values = fractions.loc[outcomes.eq(outcome), column]
            if not values.empty:
                axis.ecdf(values, color=color, linewidth=2, label=f"{outcome} (n={len(values)})")
        axis.set(
            title=algorithm, xlabel="Fração de arcos ativos (escala log)" if use_log else "Fração de arcos ativos",
            xscale="log" if use_log else "linear", ylim=(0, 1.02), yticks=np.linspace(0, 1, 6),
        )
        axis.grid()
    axes[0].set_ylabel("Proporção acumulada das demandas")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%".replace(".", ",")))
    if not valid.empty:
        axes[1].legend(loc="lower right")
    save_chart(
        figure, output_dir, "22_fracao_arcos_ativos_por_desfecho.png",
        "Fração de arcos ativos em empates e demandas com vencedor", valid, "Arcs, Arcos>0 A, Arcos>0 B, Melhor",
        "A fração divide os arcos com utilização positiva pelo total de arcos da instância.\n"
        "A altura da curva indica a proporção de demandas abaixo ou no valor do eixo horizontal.", manifest,
    )


def generate_charts(input_dir: Path, output_dir: Path) -> Manifest:
    """Gera os PNGs e o índice com a origem e o recorte de cada gráfico."""
    data, _ = read_comparisons(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Manifest = []
    with plt.rc_context(STYLE):
        plot_active_arcs(data, output_dir, manifest)
        plot_runtime_by_edges(data, output_dir, manifest)
        plot_paired_runtime(data, output_dir, manifest)
        plot_speedup_by_edges(data, output_dir, manifest)
        plot_signed_gain(
            data, output_dir, manifest, x="Arcs", xlabel="Número de arcos da instância (escala log)",
            filename="07_ganho_assinado_vs_arestas.png", title="Ganho assinado e tamanho da instância",
            fields="Arcs, Ganho, Melhor",
        )
        plot_gain_by_winner(data, output_dir, manifest)
        plot_gain_by_first_difference(data, output_dir, manifest)
        plot_first_difference_counts(data, output_dir, manifest)
        plot_waypoints_paired(data, output_dir, manifest)
        plot_waypoint_difference_vs_speedup(data, output_dir, manifest)
        plot_signed_gain(
            data, output_dir, manifest, x=WP_DELTA, xlabel="Diferença de waypoints (guloso − BRKGA)",
            filename="13_diferenca_waypoints_vs_ganho.png",
            title="Diferença de waypoints e ganho assinado",
            fields="NumWayPoints A, NumWayPoints B, Ganho, Melhor",
        )
        plot_outcomes_by_set(data, output_dir, manifest)
        plot_signed_gain(
            data, output_dir, manifest, x=ARC_DELTA, xlabel="Diferença de arcos usados (BRKGA − guloso)",
            filename="16_diferenca_arcos_vs_ganho.png",
            title="Diferença de arcos usados e ganho assinado", fields="Arcos>0 A, Arcos>0 B, Ganho, Melhor",
        )
        plot_waypoints_vs_active_arcs(data, output_dir, manifest)
        plot_tie_profiles(data, output_dir, manifest)
        plot_gain_boxplot(data, output_dir, manifest)
        plot_waypoints_by_outcome(data, output_dir, manifest)
        plot_paired_waypoint_heatmap(data, output_dir, manifest)
        plot_active_arc_fraction_by_outcome(data, output_dir, manifest)
        plot_waypoints_by_outcome(data, output_dir, manifest, split_wins=True)

    pd.DataFrame(manifest).to_csv(output_dir / "indice_graficos.csv", index=False, encoding="utf-8-sig")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos pareados a partir dos CSVs do comparador.")
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT_DIR / "comparacoes",
        help="Pasta com os arquivos set*.csv (padrão: comparacoes)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT_DIR / "graficos_comparacoes_revisados",
        help="Pasta de saída (padrão: graficos_comparacoes_revisados)",
    )
    args = parser.parse_args()

    manifest = generate_charts(args.input_dir, args.output_dir)
    print(f"{len(manifest)} gráficos em {args.output_dir.resolve()}")
    print(f"Índice de fontes e campos em {args.output_dir / 'indice_graficos.csv'}")


if __name__ == "__main__":
    main()
