"""Additional I0252 figures from saved coordinates; no simulation is run."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_landscape_010 import METHODS, LABELS, COLORS, STYLES, background, save

ROOT = Path(__file__).resolve().parent
INSTANCE = 'I0252'
EXCLUDED = {'PSoft300', 'Softmax010'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(out, surface, coordinates, rep, methods, name):
    title = f'{INSTANCE}: trayectorias, repetición {rep}'
    if len(methods) == 5:
        title += ' — cinco métodos'
    fig, ax = plt.subplots(figsize=(9, 7))
    im = background(ax, surface, title)
    for method in methods:
        q = coordinates[coordinates.method == method].sort_values('iteration')
        assert np.array_equal(q.iteration.to_numpy(), np.arange(101))
        ax.plot(q.z3, q.z4, color=COLORS[method], ls=STYLES[method],
                lw=1.5, label=LABELS[method])
        ax.scatter(q.z3.iloc[-1], q.z4.iloc[-1], s=40,
                   color=COLORS[method], edgecolor='white', zorder=10)
    start = coordinates[coordinates.iteration == 0][['z3', 'z4']]
    assert np.allclose(start.to_numpy(), [.12, .10])
    ax.scatter(.12, .10, marker='*', s=130, color='gold',
               edgecolor='black', zorder=11, label='Inicio')
    fig.colorbar(im, ax=ax, label='J(z)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.28, 1), fontsize=8)
    assert [line.get_label() for line in ax.lines] == [LABELS[m] for m in methods]
    save(fig, out, name)


def main(output):
    out = Path(output).resolve()
    source_paths = [out/'high'/f'{INSTANCE}_surface.csv']
    source_paths += sorted((out/'comparisons'/INSTANCE).glob('rep_*_coordinates.csv'))
    source_paths += sorted((out/'comparisons'/INSTANCE).glob('rep_*_seeds.csv'))
    inputs = {str(p.relative_to(out)): sha(p) for p in source_paths}
    surface = pd.read_csv(source_paths[0])
    coords = {rep: pd.read_csv(out/'comparisons'/INSTANCE/f'rep_{rep:02d}_coordinates.csv')
              for rep in range(1, 6)}
    generated = [f'{INSTANCE}_oracle_trajectories_rep{rep:02d}' for rep in range(2, 6)]
    filtered_name = f'{INSTANCE}_oracle_trajectories_rep01_five_methods'
    generated.append(filtered_name)
    # Preserve every previous figure, including the original seven-method rep01.
    protected = {p: sha(p) for p in (out/'figures').iterdir()
                 if p.suffix in ('.png', '.pdf') and p.stem not in generated}
    for rep in range(2, 6):
        draw(out, surface, coords[rep], rep, METHODS,
             f'{INSTANCE}_oracle_trajectories_rep{rep:02d}')
    kept = [method for method in METHODS if method not in EXCLUDED]
    draw(out, surface, coords[1], 1, kept, filtered_name)
    assert all(sha(out/name) == h for name, h in inputs.items())
    assert all(sha(p) == h for p, h in protected.items())

    page = ['<!doctype html><meta charset="utf-8"><title>I0252: repeticiones y cinco métodos</title>',
            '<style>body{font-family:system-ui;margin:2em;max-width:1500px}img{max-width:100%}a{margin-right:1em}</style>',
            '<h1>I0252: trayectorias sobre la superficie</h1>',
            '<p>rep01–rep05 son cinco repeticiones independientes, cada una con 100 iteraciones. '
            'Se conservan las mismas decisiones y la misma superficie estimada; sólo cambia la presentación.</p>',
            '<p><a href="index.html">Índice general</a></p>',
            '<h2>Rep01: Fixed, Fresh, Green, Softmax β=0.50 y Green+Softmax β=0.50</h2>',
            f'<p><a href="figures/{filtered_name}.png">PNG</a><a href="figures/{filtered_name}.pdf">PDF</a></p>',
            f'<img src="figures/{filtered_name}.png">',
            '<h2>Las cinco repeticiones con los siete métodos originales</h2>']
    for rep in range(1, 6):
        stem = f'{INSTANCE}_oracle_trajectories_rep{rep:02d}'
        page += [f'<h3>rep{rep:02d}</h3><p><a href="figures/{stem}.png">PNG</a>'
                 f'<a href="figures/{stem}.pdf">PDF</a></p><img src="figures/{stem}.png">']
    page_name = f'{INSTANCE}_oracle_repetitions.html'
    (out/page_name).write_text('\n'.join(page)+'\n')
    index_path = out/'index.html'
    index = index_path.read_text()
    if page_name not in index:
        link = f'<p><a href="{page_name}">I0252: trayectorias rep01–rep05 y versión de cinco métodos</a></p>\n'
        index = index.replace('<h2>', link+'<h2>', 1)
        index_path.write_text(index)
    manifest_name = 'validation/additional_oracle_plots.json'
    manifest = dict(instance=INSTANCE, generated=generated, filtered_rep=1,
                    included_methods=kept, excluded_methods=sorted(EXCLUDED),
                    source_sha256=sha(Path(__file__)), inputs_sha256=inputs,
                    source_data_unchanged=True, previous_figures_unchanged=True,
                    command='python3 plot_oracle_repetitions_011.py')
    (out/manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')
    metadata_path = out/'COMPLETE.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['png_files'] = len(list((out/'figures').glob('*.png')))
    metadata['supplemental_oracle_plots'] = manifest_name
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    shutil.copy2(Path(__file__), out/'reproduction'/Path(__file__).name)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT/'resultados/011probing_high_contrast_interior')
