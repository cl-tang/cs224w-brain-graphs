from pathlib import Path
import pandas as pd
from typing import List

def list_graphs(data_dir: str | Path) -> list[Path]:
    p = Path(data_dir)
    return sorted(p.glob("*.graphml"))

def load_pheno(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"Subject": str, "subject_id": str})
    # rename columns
    if "Subject" in df.columns: df = df.rename(columns={"Subject": "subject_id"})
    if "Age_in_Yrs" in df.columns: df = df.rename(columns={"Age_in_Yrs": "age"})
    df["subject_id"] = df["subject_id"].astype(str)
    return df[["subject_id", "age"]].copy()

def attach_age(file_paths: List[Path], pheno: pd.DataFrame) -> list[float]:
    ages = []
    idx = pheno.set_index("subject_id")["age"]
    for p in file_paths:
        sid = p.name.split("_")[0]
        ages.append(float(idx.loc[sid]))
    return ages
