import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "comparacoes"
OUTPUT_DIR = BASE_DIR / "comparacoes_simplificados"

COLUMNS_TO_REMOVE = {
    "Rota A",
    "Rota B",
    "Umax A",
    "Umax B",
    "Var. Umax",
    "Melhor",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_csv(input_file: Path):
    with input_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = [
            column
            for column in reader.fieldnames
            if column not in COLUMNS_TO_REMOVE
        ]

        rows = []
        for row in reader:
            melhor = (row.get("Melhor") or "").strip()
            
            if melhor not in ("A", "B"):
                continue

            rows.append(row)

    rows.sort(
        key=lambda row: 0 if row["Melhor"].strip() == "A" else 1
    )

    output_file = OUTPUT_DIR / f"{input_file.stem}-simplificado.csv"

    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{input_file.name} -> {output_file}")


for csv_file in sorted(INPUT_DIR.glob("set*.csv")):
    process_csv(csv_file)