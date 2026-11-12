"""Finish saved main-study artifacts after initial process ended during dimension work."""
import contextlib
import hashlib
import importlib.metadata as md
import json
import os
import pickle
import platform
import time
import pandas as pd
from threadpoolctl import threadpool_limits
from research_pipeline import OUT, ROOT, CONFIG, dimension_experiments, metrics, dump
from main import FEATURES, SimpleANFIS, choose_threshold


def run():
    start=time.perf_counter()
    df=pd.read_csv(ROOT/'data/german_credit_data.csv');y=df.target.map({'good':0,'bad':1}).to_numpy()
    assert len(pd.read_csv(OUT/'per_split_metrics.csv'))==180
    dimension_experiments(df,y)
    bundle=pickle.loads((OUT/'models/42_weighted_original.pkl').read_bytes())
    f,v,t=bundle['fit_ids'],bundle['val_ids'],bundle['test_ids'];sc=bundle['scaler']
    X=sc.transform(df[FEATURES].to_numpy(float));rows=[]
    for init in [0,1,2,3,4]:
        model=SimpleANFIS(4,learning_rate=.02,epochs=500,seed=init)
        with contextlib.redirect_stdout(None):model.fit(X[f],y[f],{0:1.,1:2.33})
        threshold=choose_threshold(y[v],model.predict_proba(X[v]));p=model.predict_proba(X[t])
        for mode,th in [('default',.5),('validation',threshold)]:
            rows.append(dict(initialization_seed=init,threshold_mode=mode,threshold=th,**metrics(y[t],p,th)))
    pd.DataFrame(rows).to_csv(OUT/'initialization_sensitivity.csv',index=False)
    dump(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        processor=platform.processor(),logical_cpus=os.cpu_count(),threads=1,wall_seconds=time.perf_counter()-start,
        wall_scope='completion run only: repeated dimensionality fits plus initialization sensitivity; initial main-study total wall time unavailable',
        packages={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','threadpoolctl']},
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['main.py','research_models.py','research_pipeline.py','reproduce_original.py','finish_research.py']},
        interruption='Initial process ended before final dimension table, initialization sensitivity or environment output was saved. Completed main-study checkpoints, scores and trials survived. Cause not confirmed. Dimension experiments were rerun in full with per-fit persistence; no numeric values inferred from incomplete log.',
        initial_mainstudy_source='executed_source/research_pipeline_initial.py',
        memory='CIM RAM query denied; workspace is explicit array estimate, not peak memory'))
    print('Completed all 30 dimension fits and 5 fixed-split initialization fits',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
