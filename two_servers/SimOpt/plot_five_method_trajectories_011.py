"""Plot all five I0252 repetitions using the requested five saved method traces."""
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

from analyze_landscape_010 import LABELS, COLORS, STYLES, save

ROOT = Path(__file__).resolve().parent
INSTANCE = 'I0252'
METHODS = ['Fixed', 'Fresh', 'Green', 'Softmax050', 'GreenSoftmax050']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(out, data, rep, variable, stem, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        q = data[data.method == method].sort_values('iteration')
        assert np.array_equal(q.iteration.to_numpy(), np.arange(101))
        assert np.isfinite(q[variable]).all()
        ax.plot(q.iteration, q[variable], color=COLORS[method],
                ls=STYLES[method], lw=1.6, label=LABELS[method])
    if variable in ('z3', 'z4'):
        ax.set(ylim=(-.025, 1.025), ylabel=variable,
               title=f'{INSTANCE}: Class-{variable[-1]} probing trajectory, rep. {rep} — cinco métodos')
    else:
        n = data.trace_paths.unique()
        assert len(n) == 1
        title = f'{INSTANCE}: convergencia, rep. {rep} — cinco métodos; evaluación CRN n={int(n[0])}'
        if zoom:
            # Retain all five curves; select the vertical range from the other
            # four methods so Fixed does not determine the zoom scale.
            values = data[data.method.isin(METHODS[1:])].objective
            pad = max(.05, .05*(values.max()-values.min()))
            ax.set_ylim(values.min()-pad, values.max()+pad)
            title += '\nZoom: rango de los métodos distintos de Fixed'
        else:
            title += '\nEscala completa'
        ax.set(ylabel='Objective value J(z)', title=title)
    ax.set(xlabel='Iteration', xlim=(0, 100))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    assert [line.get_label() for line in ax.lines] == [LABELS[m] for m in METHODS]
    save(fig, out, stem)


def main(output):
    out = Path(output).resolve()
    folder = out/'comparisons'/INSTANCE
    inputs = sorted(folder.glob('rep_*_coordinates.csv'))
    inputs += sorted(folder.glob('rep_*_traces.csv'))
    inputs += sorted(folder.glob('rep_*_seeds.csv'))
    before = {str(p.relative_to(out)): sha(p) for p in inputs}
    stems = {}
    for rep in range(1, 6):
        stems[rep] = {v: f'{INSTANCE}_{v}_trajectory_rep{rep:02d}_five_methods' for v in ('z3', 'z4')}
        stems[rep].update({v: f'{INSTANCE}_convergence_rep{rep:02d}_{v}_five_methods' for v in ('full', 'zoom')})
    generated = [name for group in stems.values() for name in group.values()]
    protected = {p: sha(p) for p in (out/'figures').iterdir()
                 if p.suffix in ('.png', '.pdf') and p.stem not in generated}
    for rep, names in stems.items():
        coordinates = pd.read_csv(folder/f'rep_{rep:02d}_coordinates.csv')
        traces = pd.read_csv(folder/f'rep_{rep:02d}_traces.csv')
        assert set(METHODS) <= set(coordinates.method) & set(traces.method)
        assert coordinates.rep.eq(rep).all() and traces.rep.eq(rep).all()
        for variable in ('z3', 'z4'):
            draw(out, coordinates, rep, variable, names[variable])
        draw(out, traces, rep, 'objective', names['full'])
        draw(out, traces, rep, 'objective', names['zoom'], zoom=True)
    assert all(sha(out/name) == h for name, h in before.items())
    assert all(sha(path) == h for path, h in protected.items())

    page = ['<!doctype html><meta charset="utf-8"><title>I0252: cinco métodos, todas las repeticiones</title>',
            '<style>body{font-family:system-ui;margin:2em;max-width:1600px}img{max-width:100%}a{margin-right:1em}td,th{padding:.6em;text-align:left}</style>',
            '<h1>I0252: z3, z4 y convergencia — cinco métodos</h1>',
            '<p>Fixed, Fresh, Green, Softmax β=0.50 y Green+Softmax β=0.50. '
            'Cinco repeticiones independientes; 101 puntos por trayectoria (iteraciones 0–100). '
            'Se reutilizan las coordenadas y evaluaciones guardadas.</p>',
            '<p>La convergencia completa incluye las cinco curvas. El zoom conserva las cinco curvas '
            'y determina su escala vertical con los métodos distintos de Fixed; parte de Fixed puede quedar fuera del rango visible.</p>',
            '<p><a href="index.html">Índice general</a><a href="I0252_oracle_repetitions.html">Trayectorias sobre la superficie</a></p>',
            '<table><tr><th>Repetición</th><th>z3</th><th>z4</th><th>Convergencia completa</th><th>Convergencia zoom</th></tr>']
    for rep, names in stems.items():
        row = f'<tr><td>rep{rep:02d}</td>'
        for kind in ('z3', 'z4', 'full', 'zoom'):
            name = names[kind]
            row += f'<td><a href="figures/{name}.png">PNG</a><a href="figures/{name}.pdf">PDF</a></td>'
        page.append(row+'</tr>')
    page.append('</table>')
    for rep, names in stems.items():
        page.append(f'<h2>rep{rep:02d}</h2>')
        for kind in ('z3', 'z4', 'full', 'zoom'):
            page.append(f'<img loading="lazy" src="figures/{names[kind]}.png">')
    page_name = f'{INSTANCE}_five_methods_repetitions.html'
    (out/page_name).write_text('\n'.join(page)+'\n')
    index_path = out/'index.html'
    index = index_path.read_text()
    if page_name not in index:
        index = index.replace('<h2>', f'<p><a href="{page_name}">I0252: z3, z4 y convergencia, cinco métodos y cinco repeticiones</a></p>\n<h2>', 1)
        index_path.write_text(index)
    manifest_name = 'validation/five_method_trajectory_plots.json'
    manifest = dict(instance=INSTANCE, repetitions=list(stems), methods=METHODS,
                    generated=generated, png_files=len(generated), pdf_files=len(generated),
                    points_per_curve=101, zoom_keeps_all_five_curves=True,
                    inputs_sha256=before, source_sha256=sha(Path(__file__)),
                    source_data_unchanged=True, previous_figures_unchanged=True,
                    command='python3 plot_five_method_trajectories_011.py')
    (out/manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')
    metadata_path = out/'COMPLETE.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['png_files'] = len(list((out/'figures').glob('*.png')))
    metadata['supplemental_five_method_trajectory_plots'] = manifest_name
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    shutil.copy2(Path(__file__), out/'reproduction'/Path(__file__).name)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT/'resultados/011probing_high_contrast_interior')
