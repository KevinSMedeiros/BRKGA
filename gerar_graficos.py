"""Generate comparison charts from the raw CSV files in ``comparacoes``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "Instancia",
    "Nodes",
    "Arcs",
    "Demanda",
    "Volume",
    "NumWayPoints A",
    "NumWayPoints B",
    "Tempo A(ms)",
    "Tempo B(ms)",
    "Speedup A/B",
    "Arcos>0 A",
    "Arcos>0 B",
    "1a dif.",
    "Ganho",
    "Melhor",
}
SET_COLORS = {
    "setA": "#2878B5",
    "setB": "#E07A24",
    "setC": "#3A9D5D",
    "setX": "#9B59B6",
}
WINNER_COLORS = {"A": "#2878B5", "B": "#E07A24"}


def parse_number(value: object) -> float:
    """Convert CSV numbers, percentages, and speedup values to floats."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?", str(value))
    if match is None:
        return np.nan
    return float(match.group(0).replace(",", "."))


def read_comparisons(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read raw comparison CSVs and validate the columns used by the charts."""
    files = sorted(input_dir.glob("set*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo set*.csv encontrado em {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} não contém as colunas: {missing_list}")

        frame["Conjunto"] = path.stem
        frame["Fonte"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "Nodes",
        "Arcs",
        "Volume",
        "NumWayPoints A",
        "NumWayPoints B",
        "Tempo A(ms)",
        "Tempo B(ms)",
        "Speedup A/B",
        "Arcos>0 A",
        "Arcos>0 B",
        "1a dif.",
    ]
    for column in numeric_columns:
        data[column] = data[column].map(parse_number)

    data["Ganho (%)"] = data["Ganho"].map(parse_number).abs()
    data["Ganho assinado (%)"] = data["Ganho (%)"].where(
        data["Melhor"].eq("A"),
        -data["Ganho (%)"],
    )
    data["Diferenca arcos (A-B)"] = data["Arcos>0 A"] - data["Arcos>0 B"]
    data["Diferenca waypoints (B-A)"] = (
        data["NumWayPoints B"] - data["NumWayPoints A"]
    )
    data["Fracao arcos ativos A"] = data["Arcos>0 A"] / data["Arcs"]
    data["Fracao arcos ativos B"] = data["Arcos>0 B"] / data["Arcs"]
    return data, [path.name for path in files]


def add_footer(
    figure: plt.Figure,
    data: pd.DataFrame,
    columns: str,
    scope: str,
) -> None:
    sources = ", ".join(sorted(data["Fonte"].dropna().unique()))
    figure.text(
        0.01,
        0.012,
        f"Fonte: comparacoes/{sources} | N={len(data)} | "
        f"Campos: {columns} | Recorte: {scope}",
        ha="left",
        va="bottom",
        fontsize=7,
        wrap=True,
    )


def save_chart(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    title: str,
    data: pd.DataFrame,
    columns: str,
    scope: str,
    manifest: list[dict[str, object]],
) -> None:
    figure.suptitle(title, fontsize=14, fontweight="bold")
    add_footer(figure, data, columns, scope)
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    manifest.append(
        {
            "arquivo": filename,
            "titulo": title,
            "fontes": ", ".join(sorted(data["Fonte"].dropna().unique())),
            "campos": columns,
            "recorte": scope,
            "registros": len(data),
        }
    )


def set_axes(axes: np.ndarray, sets: list[str]) -> list[plt.Axes]:
    flat_axes = list(np.asarray(axes, dtype=object).ravel())
    for axis, set_name in zip(flat_axes, sets):
        axis.set_title(set_name)
    for axis in flat_axes[len(sets) :]:
        axis.set_visible(False)
    return flat_axes


def set_color(set_name: str) -> str:
    return SET_COLORS.get(set_name, "#555555")


def plot_active_arcs(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    sets = sorted(data["Conjunto"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    flat_axes = set_axes(axes, sets)
    for axis, set_name in zip(flat_axes, sets):
        subset = data[data["Conjunto"].eq(set_name)]
        axis.scatter(subset["Arcos>0 A"], subset["Arcos>0 B"], alpha=0.75)
        limit = max(
            1,
            subset[["Arcos>0 A", "Arcos>0 B"]].max().max(),
        )
        axis.plot([0, limit], [0, limit], "--", color="#777777", linewidth=1)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
    for axis in flat_axes[: len(sets)]:
        axis.set_xlabel("Arcos usados por A")
        axis.set_ylabel("Arcos usados por B")
        axis.grid(alpha=0.2)
    save_chart(
        figure,
        output_dir,
        "01_arcos_usados_pareados.png",
        "Arcos usados por A e B por conjunto (pontos abaixo da diagonal: B usa menos)",
        data,
        "Arcos>0 A, Arcos>0 B",
        "Todas as demandas, incluindo empates",
        manifest,
    )


def plot_arc_difference(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    sets = sorted(data["Conjunto"].unique())
    values = [
        data.loc[data["Conjunto"].eq(set_name), "Diferenca arcos (A-B)"].dropna()
        for set_name in sets
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(values, tick_labels=sets, showmeans=True)
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_ylabel("Arcos usados por A menos arcos usados por B")
    axis.set_title("Acima de zero: A usa mais arcos; abaixo: B usa mais")
    axis.grid(axis="y", alpha=0.2)
    save_chart(
        figure,
        output_dir,
        "02_diferenca_arcos_usados.png",
        "Distribuição da diferença de arcos usados",
        data,
        "Arcos>0 A, Arcos>0 B e diferença A-B",
        "Todas as demandas, separadas por conjunto",
        manifest,
    )


def plot_active_arc_fraction(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for set_name, subset in data.groupby("Conjunto"):
        axis.scatter(
            subset["Arcs"],
            subset["Fracao arcos ativos A"],
            color=set_color(set_name),
            marker="o",
            alpha=0.65,
            label=f"{set_name} - A",
        )
        axis.scatter(
            subset["Arcs"],
            subset["Fracao arcos ativos B"],
            color=set_color(set_name),
            marker="x",
            alpha=0.65,
            label=f"{set_name} - B",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Arestas totais na instância (Arcs, escala log)")
    axis.set_ylabel("Arcos usados / arestas totais")
    axis.set_title("Fração de arcos usados em relação ao tamanho do grafo")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    save_chart(
        figure,
        output_dir,
        "03_fracao_arcos_usados.png",
        "Fração de arcos usados por tamanho do grafo",
        data,
        "Arcs, Arcos>0 A, Arcos>0 B",
        "Todas as demandas; A e B representados separadamente",
        manifest,
    )


def plot_runtime_by_edges(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    sets = sorted(data["Conjunto"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    flat_axes = set_axes(axes, sets)
    for axis, set_name in zip(flat_axes, sets):
        subset = data[data["Conjunto"].eq(set_name)]
        axis.scatter(subset["Arcs"], subset["Tempo A(ms)"], alpha=0.7, label="A")
        axis.scatter(subset["Arcs"], subset["Tempo B(ms)"], alpha=0.7, label="B")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    for axis in flat_axes[: len(sets)]:
        axis.set_xlabel("Arestas totais (Arcs, escala log)")
        axis.set_ylabel("Tempo (ms, escala log)")
    save_chart(
        figure,
        output_dir,
        "04_tempo_vs_arestas.png",
        "Tempo de execução versus número de arestas",
        data,
        "Arcs, Tempo A(ms), Tempo B(ms)",
        "Todas as demandas; escalas logarítmicas nos dois eixos",
        manifest,
    )


def plot_paired_runtime(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for set_name, subset in data.groupby("Conjunto"):
        axis.scatter(
            subset["Tempo A(ms)"],
            subset["Tempo B(ms)"],
            color=set_color(set_name),
            alpha=0.65,
            label=set_name,
        )
    positive = data[["Tempo A(ms)", "Tempo B(ms)"]].where(
        data[["Tempo A(ms)", "Tempo B(ms)"]].gt(0)
    ).stack()
    if not positive.empty:
        low, high = positive.min(), positive.max()
        axis.plot([low, high], [low, high], "--", color="#555555", label="A = B")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Tempo de A (ms, escala log)")
    axis.set_ylabel("Tempo de B (ms, escala log)")
    axis.set_title("Pontos abaixo da diagonal indicam menor tempo de B")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "05_tempo_A_vs_B.png",
        "Comparação pareada dos tempos de execução",
        data,
        "Tempo A(ms), Tempo B(ms)",
        "Todas as demandas, incluindo empates",
        manifest,
    )


def plot_speedup_by_edges(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    sets = sorted(data["Conjunto"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    flat_axes = set_axes(axes, sets)
    for axis, set_name in zip(flat_axes, sets):
        subset = data[data["Conjunto"].eq(set_name)]
        axis.scatter(subset["Arcs"], subset["Speedup A/B"], alpha=0.75)
        axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in flat_axes[: len(sets)]:
        axis.set_xlabel("Arestas totais (Arcs, escala log)")
        axis.set_ylabel("Tempo A / tempo B (escala log)")
    save_chart(
        figure,
        output_dir,
        "06_speedup_vs_arestas.png",
        "Speedup A/B por tamanho da instância (acima de 1: B mais rápido)",
        data,
        "Arcs, Speedup A/B",
        "Todas as demandas; speedup acima de 1 significa Tempo A > Tempo B",
        manifest,
    )


def plot_signed_gain_by_edges(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    decided = data[data["Melhor"].isin(["A", "B"])]
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset["Arcs"],
            subset["Ganho assinado (%)"],
            color=WINNER_COLORS[winner],
            alpha=0.7,
            label=f"Vence {winner}",
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Arestas totais (Arcs, escala log)")
    axis.set_ylabel("Ganho assinado (%) (positivo: A; negativo: B)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "07_ganho_assinado_vs_arestas.png",
        "Ganho no primeiro critério divergente versus número de arestas",
        decided,
        "Arcs, Ganho, Melhor",
        "Somente demandas decididas (A ou B); empates não têm ganho",
        manifest,
    )


def plot_gain_by_winner(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[data["Melhor"].isin(["A", "B"]) & data["Ganho (%)"].notna()]
    sets = sorted(data["Conjunto"].unique())
    columns = min(2, len(sets))
    rows = (len(sets) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 4.5 * rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.ravel())
    for axis, set_name in zip(flat_axes, sets):
        subset = decided[decided["Conjunto"].eq(set_name)]
        counts = subset["Melhor"].value_counts()
        values = [
            subset.loc[subset["Melhor"].eq(winner), "Ganho (%)"].dropna()
            for winner in ("A", "B")
        ]
        labels = [
            f"Vence {winner} (n={counts.get(winner, 0)})"
            for winner in ("A", "B")
        ]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.set_title(set_name)
        axis.grid(axis="y", alpha=0.2)
    for axis in flat_axes[len(sets) :]:
        axis.set_visible(False)
    flat_axes[0].set_ylabel("Magnitude do ganho no primeiro critério divergente (%)")
    save_chart(
        figure,
        output_dir,
        "08_distribuicao_ganho_por_vencedor.png",
        "Distribuição do ganho, separada pelo vencedor e pelo conjunto",
        decided,
        "Ganho, Melhor",
        "Somente demandas decididas; magnitude do ganho, separada por vencedor",
        manifest,
    )


def plot_gain_by_first_difference(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[
        data["Melhor"].isin(["A", "B"])
        & data["Ganho (%)"].notna()
        & data["1a dif."].notna()
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset["1a dif."],
            subset["Ganho (%)"],
            color=WINNER_COLORS[winner],
            alpha=0.7,
            label=f"Vence {winner}",
        )
    axis.set_xlabel("Posição da primeira diferença (índice começando em 0)")
    axis.set_ylabel("Magnitude do ganho nessa posição (%)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "09_ganho_vs_primeira_diferenca.png",
        "Ganho versus posição da primeira diferença no vetor de fitness",
        decided,
        "1a dif., Ganho, Melhor",
        "Somente demandas decididas com primeira diferença registrada",
        manifest,
    )


def plot_first_difference_counts(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[data["Melhor"].isin(["A", "B"]) & data["1a dif."].notna()].copy()
    decided["1a dif."] = decided["1a dif."].astype(int)
    counts = pd.crosstab(decided["1a dif."], decided["Melhor"]).sort_index()
    counts = counts.reindex(columns=["A", "B"], fill_value=0)
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot(
        kind="bar",
        stacked=True,
        color=[WINNER_COLORS["A"], WINNER_COLORS["B"]],
        ax=axis,
    )
    axis.set_xlabel("Posição da primeira diferença (índice começando em 0)")
    axis.set_ylabel("Número de demandas")
    axis.set_title("Contagem por vencedor; empates excluídos")
    axis.legend(title="Vencedor")
    axis.grid(axis="y", alpha=0.2)
    save_chart(
        figure,
        output_dir,
        "10_primeira_diferenca_por_vencedor.png",
        "Frequência da primeira diferença por posição e vencedor",
        decided,
        "1a dif., Melhor",
        "Somente demandas decididas com primeira diferença registrada",
        manifest,
    )


def plot_waypoints_paired(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    sets = sorted(data["Conjunto"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    flat_axes = set_axes(axes, sets)
    for axis, set_name in zip(flat_axes, sets):
        subset = data[data["Conjunto"].eq(set_name)]
        axis.scatter(
            subset["NumWayPoints A"], subset["NumWayPoints B"], alpha=0.75
        )
        limit = max(
            1,
            subset[["NumWayPoints A", "NumWayPoints B"]].max().max(),
        )
        axis.plot([0, limit], [0, limit], "--", color="#777777", linewidth=1)
        axis.grid(alpha=0.2)
    for axis in flat_axes[: len(sets)]:
        axis.set_xlabel("Waypoints de A")
        axis.set_ylabel("Waypoints de B")
    save_chart(
        figure,
        output_dir,
        "11_waypoints_A_vs_B.png",
        "Número de waypoints nas soluções de A e B",
        data,
        "NumWayPoints A, NumWayPoints B",
        "Todas as demandas, incluindo empates; linha diagonal indica valores iguais",
        manifest,
    )


def plot_waypoint_difference_vs_speedup(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for set_name, subset in data.groupby("Conjunto"):
        axis.scatter(
            subset["Diferenca waypoints (B-A)"],
            subset["Speedup A/B"],
            color=set_color(set_name),
            alpha=0.65,
            label=set_name,
        )
    axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Waypoints de B menos waypoints de A")
    axis.set_ylabel("Tempo A / tempo B (escala log; acima de 1: B mais rápido)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "12_diferenca_waypoints_vs_speedup.png",
        "Diferença de waypoints versus speedup",
        data,
        "NumWayPoints A, NumWayPoints B, Speedup A/B",
        "Todas as demandas; diferença positiva significa mais waypoints em B",
        manifest,
    )


def plot_waypoint_difference_vs_gain(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[data["Melhor"].isin(["A", "B"]) & data["Ganho (%)"].notna()]
    figure, axis = plt.subplots(figsize=(9, 6))
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset["Diferenca waypoints (B-A)"],
            subset["Ganho assinado (%)"],
            color=WINNER_COLORS[winner],
            alpha=0.7,
            label=f"Vence {winner}",
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Waypoints de B menos waypoints de A")
    axis.set_ylabel("Ganho assinado (%) (positivo: A; negativo: B)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "13_diferenca_waypoints_vs_ganho.png",
        "Diferença de waypoints versus ganho assinado",
        decided,
        "NumWayPoints A, NumWayPoints B, Ganho, Melhor",
        "Somente demandas decididas; ganho positivo favorece A",
        manifest,
    )


def plot_outcomes_by_set(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    counts = pd.crosstab(data["Conjunto"], data["Melhor"])
    counts = counts.reindex(columns=["A", "B", "Empate"], fill_value=0)
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot(
        kind="bar",
        stacked=True,
        color=[WINNER_COLORS["A"], WINNER_COLORS["B"], "#999999"],
        ax=axis,
    )
    axis.set_xlabel("Conjunto de instâncias")
    axis.set_ylabel("Número de demandas")
    axis.set_title("Inclui empates; dados dos CSVs originais")
    axis.legend(title="Melhor")
    axis.grid(axis="y", alpha=0.2)
    save_chart(
        figure,
        output_dir,
        "14_resultados_por_conjunto.png",
        "Vitórias e empates por conjunto",
        data,
        "Conjunto, Melhor",
        "Todas as demandas, incluindo empates",
        manifest,
    )


def plot_gain_vs_volume(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[
        data["Melhor"].isin(["A", "B"])
        & data["Ganho (%)"].notna()
        & data["Volume"].gt(0)
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset["Volume"],
            subset["Ganho assinado (%)"],
            color=WINNER_COLORS[winner],
            alpha=0.7,
            label=f"Vence {winner}",
        )
    axis.set_xscale("log")
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("Volume (escala log)")
    axis.set_ylabel("Ganho assinado (%) (positivo: A; negativo: B)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "15_ganho_vs_volume.png",
        "Ganho assinado versus volume",
        decided,
        "Volume, Ganho, Melhor",
        "Somente demandas decididas com volume positivo",
        manifest,
    )


def plot_arc_difference_vs_gain(
    data: pd.DataFrame, output_dir: Path, manifest: list[dict[str, object]]
) -> None:
    decided = data[data["Melhor"].isin(["A", "B"]) & data["Ganho (%)"].notna()]
    figure, axis = plt.subplots(figsize=(9, 6))
    for winner, subset in decided.groupby("Melhor"):
        axis.scatter(
            subset["Diferenca arcos (A-B)"],
            subset["Ganho assinado (%)"],
            color=WINNER_COLORS[winner],
            alpha=0.7,
            label=f"Vence {winner}",
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
    axis.set_xlabel("Arcos usados por A menos arcos usados por B")
    axis.set_ylabel("Ganho assinado (%) (positivo: A; negativo: B)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_chart(
        figure,
        output_dir,
        "16_diferenca_arcos_vs_ganho.png",
        "Diferença de arcos usados versus ganho assinado",
        decided,
        "Arcos>0 A, Arcos>0 B, Ganho, Melhor",
        "Somente demandas decididas; ganho positivo favorece A",
        manifest,
    )


def generate_charts(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    """Load data and generate all charts, returning their data-source manifest."""
    data, _ = read_comparisons(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    chart_functions = (
        plot_active_arcs,
        plot_arc_difference,
        plot_active_arc_fraction,
        plot_runtime_by_edges,
        plot_paired_runtime,
        plot_speedup_by_edges,
        plot_signed_gain_by_edges,
        plot_gain_by_winner,
        plot_gain_by_first_difference,
        plot_first_difference_counts,
        plot_waypoints_paired,
        plot_waypoint_difference_vs_speedup,
        plot_waypoint_difference_vs_gain,
        plot_outcomes_by_set,
        plot_gain_vs_volume,
        plot_arc_difference_vs_gain,
    )
    for chart_function in chart_functions:
        chart_function(data, output_dir, manifest)

    pd.DataFrame(manifest).to_csv(
        output_dir / "indice_graficos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de comparação a partir dos CSVs originais."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "comparacoes",
        help="Pasta com os arquivos set*.csv (padrão: comparacoes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "graficos_comparacoes",
        help="Pasta de saída (padrão: graficos_comparacoes na raiz)",
    )
    args = parser.parse_args()

    manifest = generate_charts(args.input_dir, args.output_dir)
    print(f"Gráficos gerados: {len(manifest)}")
    print(f"Saída: {args.output_dir.resolve()}")
    print(f"Índice de fontes e campos: {args.output_dir / 'indice_graficos.csv'}")


if __name__ == "__main__":
    main()
