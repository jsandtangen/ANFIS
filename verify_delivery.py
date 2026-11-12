"""Check delivered artifact consistency and create a complete SHA256 manifest."""
import csv
import hashlib
import json
import pickle
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_pipeline import predict, split

OUT=Path('results/research')


def main():
    with (OUT/'tests.log').open('w',encoding='utf-8') as log:
        outcome=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_research'))
    (OUT/'test_outcome.json').write_text(json.dumps(dict(tests_run=outcome.testsRun,
        failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),
        successful=outcome.wasSuccessful()),indent=2))
    assert outcome.wasSuccessful() and not outcome.skipped
    documents=['RESEARCH_AUDIT.md','REPRODUCTION_REPORT.md','EXPERIMENTAL_DESIGN.md',
        'ANFIS_DIAGNOSTICS.md','MODEL_COMPARISON.md','INTERPRETABILITY_CASE_STUDIES.md',
        'RULE_COMPLEXITY_ANALYSIS.md','REPORT_READY_RESULTS.md','LIMITATIONS_AND_DISCUSSION.md','README.md']
    for name in documents:assert (OUT/name).is_file(),name
    for name in ['main.py','requirements.txt']:
        assert Path(name).read_bytes()==(OUT/'original_snapshot'/name).read_bytes(),name
    for path in (OUT/'original_snapshot').glob('*.csv'):
        assert path.read_bytes()==(Path('results')/path.name).read_bytes(),path
    assert (OUT/'original_snapshot/anfis_rules.txt').read_bytes()==Path('results/anfis_rules.txt').read_bytes()
    df=pd.read_csv('data/german_credit_data.csv');y=df.target.map({'good':0,'bad':1}).to_numpy()
    scores=pd.read_csv(OUT/'test_predictions.csv');maximum=0.;scaler_error=0.
    checkpoints=list((OUT/'models').glob('*.pkl'));assert len(checkpoints)==90
    for path in checkpoints:
        with path.open('rb') as f:bundle=pickle.load(f)
        model=bundle['model'];seed=bundle['seed'];f,v,t=split(y,seed)
        np.testing.assert_array_equal(f,bundle['fit_ids']);np.testing.assert_array_equal(v,bundle['val_ids'])
        np.testing.assert_array_equal(t,bundle['test_ids'])
        raw=df[bundle['features']].to_numpy(float)
        err=float(np.max(np.abs(bundle['scaler'].mean_-raw[f].mean(axis=0))))
        scaler_error=max(scaler_error,err)
        condition=path.stem.split('_')[1];name=path.stem.split('_',2)[2]
        sub=scores[(scores.seed==seed)&(scores.condition==condition)&(scores.model==name)].set_index('row_id').loc[t]
        p=np.repeat(y[f].mean(),len(t)) if model is None else predict(model,bundle['scaler'].transform(raw[t]))
        error=float(np.max(np.abs(p-sub.score.to_numpy())));maximum=max(maximum,error)
        assert error<1e-12,(path,error)
        if getattr(model,'trainable',False):
            assert np.isfinite(model.centers).all() and np.isfinite(model.coef).all()
            assert (np.exp(model.log_widths)>=.05-1e-12).all() and (np.exp(model.log_widths)<=5+1e-12).all()
    assert scaler_error<1e-9
    dim=pd.read_csv(OUT/'dimension_metrics.csv');assert len(dim)==30
    assert np.array_equal(dim.rules,3**dim.d)
    assert np.array_equal(dim.consequent_parameters,3**dim.d*(dim.d+1))
    convergence=pd.read_csv(OUT/'dimension_convergence.csv');assert len(convergence)==30
    assert len(list((OUT/'dimension_replay').glob('*.pkl')))==30
    assert len(list((OUT/'dimension_replay').glob('*_trajectory.csv')))==30
    result=dict(status='passed',required_documents=len(documents),checkpoints_recomputed=len(checkpoints),
        maximum_score_difference=maximum,maximum_scaler_mean_difference=scaler_error,dimension_rows=len(dim),
        original_source_and_result_bytes_preserved=True,dimension_convergence_replays=30,
        notes='These consistency checks do not validate missing report/exam compliance or prove absence of every possible design error.')
    (OUT/'delivery_verification.json').write_text(json.dumps(result,indent=2))
    sources=['main.py','reproduce_original.py','research_models.py','research_pipeline.py','research_telemetry.py',
        'build_research_reports.py','build_final_summary.py','finish_research.py','dimension_convergence.py',
        'verify_delivery.py','test_research.py','README.md','Run.md',
        'requirements.txt','requirements-research.txt','requirements-research.lock.txt','data/german_credit_data.csv']
    if Path('FINAL_RESEARCH_SUMMARY.md').exists():sources.append('FINAL_RESEARCH_SUMMARY.md')
    paths=[Path(p) for p in sources]+[p for p in OUT.rglob('*') if p.is_file() and p.name!='FILE_MANIFEST.csv']
    with (OUT/'FILE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader()
        for path in sorted(set(paths)):
            writer.writerow(dict(path=path.as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
