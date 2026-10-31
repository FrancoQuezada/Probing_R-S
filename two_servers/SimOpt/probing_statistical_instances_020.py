"""Canonical instance selection and exact historical parameter provenance."""
import csv
from pathlib import Path

import toml

from i0252_statistical_storage_019 import ROOT, DEFAULT_OUTPUT, sha

INSTANCES = ("L000", "L025", "L026", "L030")
BASE_OUTPUT = ROOT / "resultados/020probing_four_instances_statistical"
MODEL_SOURCE = ROOT / "resultados/010probing_instance_landscape_search/configs/models.toml"


def canonical_instance(value):
    value = value.upper()
    if value == "L30":
        value = "L030"
    if value not in (*INSTANCES, "I0252"):
        raise ValueError(f"Unknown instance: {value}")
    return value


def default_output(instance):
    instance = canonical_instance(instance)
    return DEFAULT_OUTPUT if instance == "I0252" else BASE_OUTPUT / instance


def original_instance(instance):
    """Read source tables, cross-check 011 and archived runs; never infer parameters."""
    instance = canonical_instance(instance)
    if instance == "I0252":
        raise ValueError("I0252 retains its original initializer")
    source010 = MODEL_SOURCE.parent.parent
    source011 = ROOT / "resultados/011probing_high_contrast_interior"
    entry = toml.load(MODEL_SOURCE)[instance]
    model = {key: value for key, value in entry.items() if key.isupper()}
    if entry != toml.load(source011 / "configs/reference010_models.toml")[instance]:
        raise ValueError(f"010/011 reference table mismatch: {instance}")
    controls = toml.load(source011 / "configs/models.toml")
    if instance in controls and model != controls[instance]:
        raise ValueError(f"011 effective model mismatch: {instance}")
    science = toml.load(source010 / "configs/experiment.toml")["science"]
    expected = {"Z0": [.12, .10], "BASE_STEP": .035, "GRAD_CLIP": 15., "max_iters": 100,
                "n_paths": 500, "jobs": 45, "trace_paths": 1000, "selection_paths": 10000, "final_paths": 50000}
    for key, value in expected.items():
        if science[key] != value:
            raise ValueError(f"Historical optimization protocol differs: {instance}/{key}")
    coordinates = source010 / f"comparisons/{instance}/rep_01_coordinates.csv"
    with open(coordinates, newline="") as file:
        initial = next(csv.DictReader(file))
    if [float(initial[key]) for key in ("z3", "z4")] != science["Z0"]:
        raise ValueError(f"Archived initial point mismatch: {instance}")
    surface = source010 / f"high/{instance}_surface.csv"
    best = None
    if surface.exists():
        with open(surface, newline="") as file:
            surface_rows = list(csv.DictReader(file))
        if any(row["scenario"] != instance for row in surface_rows):
            raise ValueError(f"Wrong reference surface: {instance}")
        best = max(surface_rows, key=lambda row: float(row["objective"]))
    return dict(instance=instance, model=model, original_entry=entry, science=science,
                model_sha256=entry["model_sha256"], source=MODEL_SOURCE,
                source_sha256=sha(MODEL_SOURCE), policies=source010 / "configs/policies.csv",
                surface=surface if surface.exists() else None,
                reference_grid_best=None if best is None else [float(best["z3"]), float(best["z4"])],
                cross_checks=["010 models.toml", "011 reference010_models.toml", "010 archived initial point"] +
                             (["011 effective models.toml"] if instance in controls else []))
