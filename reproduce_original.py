"""Execute unchanged original main with outputs redirected; preserve existing artifacts."""
import contextlib
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import importlib.metadata as md
import matplotlib
matplotlib.use('Agg')
from threadpoolctl import threadpool_limits
import main

OUT = Path('results/research/original_reproduction')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    main.RESULTS_DIR = OUT
    main.FIGURES_DIR = OUT / 'figures'
    main.FIGURES_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    with threadpool_limits(limits=1), (OUT/'execution.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            main.main()
    meta = dict(elapsed_seconds=time.perf_counter()-start, python=platform.python_version(),
                platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                packages={p: md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']},
                main_sha256=hashlib.sha256(Path('main.py').read_bytes()).hexdigest(),
                data_sha256=hashlib.sha256(main.DATA_PATH.read_bytes()).hexdigest(),
                hardware_note='Win32 CIM processor/RAM queries denied; processor string and logical count available.',
                threads=1)
    (OUT/'environment.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    run()
