"""Portable, non-destructive reproduction entry point (Python 3.11+)."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, urllib.request, zipfile

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'package.json').read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf8')


def run(args, cwd=ROOT):
    print('RUN', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def verify(require_research=False):
    checked = 0
    for name in ['tracked_files.json', 'research_files.json']:
        inventory = json.loads((ROOT / 'manifests' / name).read_text(encoding='utf8'))
        if name.startswith('research') and not (ROOT / 'research').exists():
            if require_research:
                raise RuntimeError('Download the research asset first: python reproduce.py download')
            print('Research archive not installed; checking the self-contained paper replay only.')
            continue
        for relative, spec in inventory.items():
            path = ROOT / relative
            if not path.is_file() or path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
                raise RuntimeError(f'Integrity mismatch: {relative}')
            checked += 1
    report = {'status': 'PASS', 'files_checked': checked, 'research_present': (ROOT / 'research').exists()}
    save(ROOT / 'runs' / 'integrity.json', report)
    print(json.dumps(report))
    return report


def download():
    if (ROOT / 'research').exists():
        return verify(True)
    spec = json.loads((ROOT / 'manifests' / 'release_assets.json').read_text(encoding='utf8'))['research']
    dest = ROOT / 'downloads' / spec['name']
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists() or sha(dest) != spec['sha256']:
        temp = dest.with_suffix('.part')
        print('Downloading', spec['url'], flush=True)
        req = urllib.request.Request(spec['url'], headers={'User-Agent': 'pelleting-reproducibility-v3'})
        with urllib.request.urlopen(req, timeout=120) as src, temp.open('wb') as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
        if sha(temp) != spec['sha256']:
            raise RuntimeError('Download failed SHA-256 verification; archive has not been extracted.')
        temp.replace(dest)
    with zipfile.ZipFile(dest) as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            if not target.is_relative_to((ROOT / 'research').resolve()):
                raise RuntimeError('Unsafe archive path: ' + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('Symbolic links are not accepted in research assets.')
        archive.extractall(ROOT)
    return verify(True)


def metrics():
    import numpy as np
    import pandas as pd
    data = ROOT / 'analysis' / 'data'
    rows = []
    if CONFIG['kind'] == 'real':
        test = pd.read_csv(data / 'prueba_original.csv')
        for model in ['Ridge', 'SEC']:
            e = test[model].to_numpy() - test.energia_peletizado_kWh.to_numpy()
            rows.append({'scope': 'final_test', 'model': model, 'n': len(e), 'MAE': abs(e).mean(),
                         'RMSE': np.sqrt((e * e).mean()), 'bias': e.mean()})
        np.testing.assert_allclose(rows[0]['MAE'], 52.448851, atol=1e-5, rtol=0)
        np.testing.assert_allclose(rows[1]['MAE'], 61.287196, atol=1e-5, rtol=0)
        pred = pd.read_csv(data / 'gpu' / 'predictions.csv')
        expected = pd.read_csv(data / 'gpu' / 'metrics.csv').set_index('modelo')
        for model, group in pred.groupby('modelo'):
            e = group.prediccion_kWh.to_numpy() - group.energia_peletizado_kWh.to_numpy()
            row = {'scope': 'temporal_development', 'model': model, 'n': len(e), 'MAE': abs(e).mean(),
                   'RMSE': np.sqrt((e * e).mean()), 'bias': e.mean()}
            np.testing.assert_allclose([row['MAE'], row['RMSE']], expected.loc[model, ['MAE_kWh', 'RMSE_kWh']].to_numpy(float), rtol=1e-10, atol=1e-8)
            rows.append(row)
        for level in [90, 95]:
            covered = test.energia_peletizado_kWh.between(test[f'L{level}'], test[f'U{level}'])
            rows.append({'scope': 'fixed_intervals', 'model': 'Ridge', 'level': level, 'n': len(test),
                         'coverage': covered.mean(), 'mean_width': (test[f'U{level}'] - test[f'L{level}']).mean()})
    else:
        pack = np.load(data / 'common_predictions.npz', allow_pickle=False)
        truth = pack['truth']
        expected = pd.read_csv(data / 'gpu_metrics.csv')
        for model in CONFIG['models']:
            pred = pack[model]
            for h in [1, 6, 30]:
                for target in ['power', 'energy']:
                    yt, yp = (truth[:, h-1], pred[:, h-1]) if target == 'power' else (truth[:, :h].sum(axis=1)/360, pred[:, :h].sum(axis=1)/360)
                    e = yp - yt
                    row = {'scope': 'common_test', 'model': model, 'h_steps': h, 'target': target,
                           'n': len(e), 'MAE': abs(e).mean(), 'RMSE': np.sqrt((e * e).mean()), 'bias': e.mean()}
                    ref = expected[(expected.model == model) & (expected.h_steps == h) & (expected.target == target) & (expected.stratum == 'all')]
                    if len(ref) != 1:
                        raise RuntimeError(f'Missing unique reference for {model}/{h}/{target}')
                    np.testing.assert_allclose([row['MAE'], row['RMSE']], ref[['MAE', 'RMSE']].iloc[0], rtol=1e-6, atol=1e-6)
                    rows.append(row)
        fullpath = ROOT / 'research' / 'temporal_comparacion_v1' / 'predictions_test.npz'
        if fullpath.exists():
            full = np.load(fullpath, allow_pickle=False)
            expected = pd.read_csv(data / 'full_metrics.csv')
            for model in ['Persistence', 'Ridge', 'GB', 'TCN']:
                for h in [1, 6, 30]:
                    for target in ['power', 'energy']:
                        yt, yp = (full['truth'][:, h-1], full[model][:, h-1]) if target == 'power' else (full['truth'][:, :h].sum(axis=1)/360, full[model][:, :h].sum(axis=1)/360)
                        e = yp - yt
                        row = {'scope': 'full_original_test', 'model': model, 'h_steps': h, 'target': target,
                               'n': len(e), 'MAE': abs(e).mean(), 'RMSE': np.sqrt((e * e).mean()), 'bias': e.mean()}
                        ref = expected[(expected.model == model) & (expected.h_steps == h) & (expected.target == target)]
                        np.testing.assert_allclose([row['MAE'], row['RMSE']], ref[['MAE','RMSE']].iloc[0], rtol=1e-6, atol=1e-6)
                        rows.append(row)
    dest = ROOT / 'runs' / 'replay'
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / 'recomputed_metrics.csv', index=False)
    save(dest / 'metrics_verification.json', {'status': 'PASS', 'comparisons': len(rows),
                                            'source': 'frozen predictions; no model fitting or model selection'})
    print('Metrics verified:', len(rows), 'rows')


def figures():
    stage = ROOT / 'runs' / 'replay' / 'analysis'
    shutil.copytree(ROOT / 'analysis', stage, dirs_exist_ok=True)
    run([sys.executable, '-X', 'utf8', stage / 'make_figures.py'])
    destination = ROOT / 'runs' / 'replay' / 'latex'
    shutil.copytree(ROOT / 'latex', destination, dirs_exist_ok=True)
    for folder in ['figures', 'tables']:
        if (stage / folder).exists():
            shutil.copytree(stage / folder, destination / folder, dirs_exist_ok=True)
    print('Regenerated figures:', stage / 'figures')


def paper():
    stage = ROOT / 'runs' / 'replay' / 'latex'
    if not (stage / 'main.tex').exists():
        shutil.copytree(ROOT / 'latex', stage, dirs_exist_ok=True)
    for exe, args in [('pdflatex', ['-interaction=nonstopmode','-halt-on-error','main.tex']),
                      ('bibtex', ['main']), ('pdflatex', ['-interaction=nonstopmode','-halt-on-error','main.tex']),
                      ('pdflatex', ['-interaction=nonstopmode','-halt-on-error','main.tex'])]:
        binary = shutil.which(exe)
        if not binary:
            raise RuntimeError(f'{exe} is required (TeX Live or MiKTeX).')
        result = subprocess.run([binary, *args], cwd=stage, capture_output=True, text=True, encoding='utf8', errors='replace')
        if result.returncode:
            raise RuntimeError(result.stdout[-7000:] + result.stderr[-2000:])
    log = (stage / 'main.log').read_text(encoding='utf8', errors='replace')
    problems = [t for t in ['undefined references','undefined citations','Overfull \\hbox','Missing character:'] if t in log]
    if problems:
        raise RuntimeError('Review LaTeX log: ' + ', '.join(problems))
    target = stage / CONFIG['pdf']
    shutil.copyfile(stage / 'main.pdf', target)
    from pypdf import PdfReader
    old, new = PdfReader(ROOT / 'latex' / CONFIG['pdf']), PdfReader(target)
    if len(old.pages) != len(new.pages):
        raise RuntimeError('PDF page count changed; inspect rendering.')
    save(ROOT / 'runs' / 'replay' / 'pdf_verification.json', {'status': 'PASS', 'pages': len(new.pages),
         'original_sha256': sha(ROOT/'latex'/CONFIG['pdf']), 'rebuilt_sha256': sha(target),
         'note': 'Build metadata and regenerated figure metadata may change PDF bytes. Original delivery is preserved.'})
    print('Compiled:', target)


def fetch_models(destination):
    from huggingface_hub import snapshot_download
    cache = Path(destination).resolve() / 'modelos_avanzados_v1' / 'hf_cache' / 'hub'
    for spec in CONFIG['foundations'].values():
        print('Fetching immutable model revision:', spec['repo'], spec['revision'], flush=True)
        snapshot_download(repo_id=spec['repo'], revision=spec['revision'], cache_dir=str(cache))


def diagnostics():
    """Re-derive all synthetic figure data from complete frozen predictions."""
    if CONFIG['kind'] != 'synthetic':
        raise RuntimeError('Dense diagnostics apply to the synthetic study; use metrics for real records.')
    verify(True)
    stage = ROOT / 'runs' / 'diagnostics'
    if stage.exists():
        raise RuntimeError('runs/diagnostics already exists; preserve it and use a fresh repository copy to repeat.')
    shutil.copytree(ROOT / 'research', stage)
    out = stage / 'paper_latex_v3'
    (out / 'data').mkdir(parents=True, exist_ok=True)
    run([sys.executable, '-X', 'utf8', out / 'prepare_diagnostics.py'])
    import pandas as pd
    files = ['first_day_native.csv','common_metadata.csv','dense_metrics.csv','error_quantiles.csv','trajectory_cases.csv','trajectory_metadata.csv']
    for name in files:
        old = pd.read_csv(ROOT / 'analysis' / 'data' / name)
        new = pd.read_csv(out / 'data' / name)
        pd.testing.assert_frame_equal(old, new, check_exact=False, rtol=1e-10, atol=1e-10)
    save(stage / 'diagnostics_reproduction.json', {'status':'PASS','compared_tables':files,
          'n_dense_metric_rows':len(pd.read_csv(out/'data/dense_metrics.csv')),'new_model_fits':0})


def prepare_training(recipe, destination):
    verify(True)
    dest = Path(destination).resolve()
    if dest.exists():
        raise RuntimeError('Choose a NEW output directory; no experiment is overwritten.')
    if not dest.is_relative_to((ROOT / 'runs').resolve()):
        raise RuntimeError('Fresh training runs must be inside this repository runs/ directory.')
    allowed = CONFIG['recipes']
    if recipe not in allowed:
        raise RuntimeError('Choose recipe: ' + ', '.join(allowed))
    spec = allowed[recipe]
    clean = set(spec['reset_folders'])
    def ignore(directory, names):
        rel = Path(directory).relative_to(ROOT / 'research').as_posix()
        if rel in clean:
            return [n for n in names if not (n.endswith('.py') or n in spec.get('keep_files', []))]
        return [n for n in names if n in ['hf_cache', '__pycache__']]
    shutil.copytree(ROOT / 'research', dest, ignore=ignore)
    # Portability only: do not require the original virtual environment directory name.
    patches = []
    for script in (dest / 'modelos_gpu_v1').glob('*.py') if (dest / 'modelos_gpu_v1').exists() else []:
        before = script.read_text(encoding='utf8')
        after = before.replace(" and 'modelos_gpu_v1' in torch.__file__", '')
        after = '\n'.join(line for line in after.split('\n') if not (line.strip().startswith('assert ') and "'modelos_gpu_v1' in torch.__file__" in line))
        if after != before:
            original = sha(script)
            script.write_text(after, encoding='utf8')
            patches.append({'path': script.relative_to(dest).as_posix(), 'reason': 'Remove environment-directory-name assertion; CUDA requirement preserved', 'before': original, 'after': sha(script)})
    commands = [[sys.executable, '-X', 'utf8', *command] for command in spec['commands']]
    plan = {'recipe': recipe, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'commands': spec['commands'], 'portability_patches': patches, 'executed': False,
            'meaning': spec['meaning'], 'code_sha256': {p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*.py')}}
    save(dest / 'REPRODUCTION_PLAN.json', plan)
    launch = '''from pathlib import Path
import json, os, subprocess, sys, hashlib
root=Path(__file__).resolve().parent
plan=json.loads((root/'REPRODUCTION_PLAN.json').read_text(encoding='utf8'))
for rel, expected in plan['code_sha256'].items():
    with (root/rel).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected: raise RuntimeError('Code changed: '+rel)
env=os.environ.copy()
env['HF_HOME']=str(root/'modelos_avanzados_v1/hf_cache')
env['HF_HUB_OFFLINE']='1'
for command in plan['commands']:
    print('RUN',command,flush=True)
    subprocess.run([sys.executable,'-X','utf8',*command],cwd=root,env=env,check=True)
plan['executed']=True
(root/'REPRODUCTION_COMPLETE.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
'''
    (dest / 'execute.py').write_text(launch, encoding='utf8')
    print('Prepared:', dest)
    print('Recipe:', spec['meaning'])
    print('For foundation models first run: python reproduce.py fetch-models --destination', dest)
    print('Then execute:', sys.executable, dest / 'execute.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify','download','metrics','figures','paper','all','diagnostics','fetch-models','prepare-training'])
    p.add_argument('--recipe')
    p.add_argument('--destination')
    args = p.parse_args()
    if args.command == 'all':
        verify(); metrics(); figures(); paper()
    elif args.command == 'fetch-models':
        fetch_models(args.destination or ROOT / 'research')
    elif args.command == 'prepare-training':
        if not args.recipe or not args.destination:
            p.error('--recipe and --destination are required')
        prepare_training(args.recipe, args.destination)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
