"""Deterministic convergence replay, checked against completed dimension metrics.

Original fixed-premise GD is cached/vectorized. Runtime here is NOT the original
implementation benchmark. Fresh model states and trajectories are diagnostic replays.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from main import SimpleANFIS,choose_threshold
from research_models import ResearchANFIS
from research_pipeline import OUT,split,metrics


def run():
    df=pd.read_csv('data/german_credit_data.csv');y=df.target.map({'good':0,'bad':1}).to_numpy()
    completed=pd.read_csv(OUT/'dimension_metrics.csv');assert len(completed)==30
    dest=OUT/'dimension_replay';dest.mkdir(exist_ok=True)
    convergence=[]
    for _,row in completed.iterrows():
        features=row.features.split(';');seed=int(row.seed);f,v,t=split(y,seed)
        sc=StandardScaler().fit(df[features].to_numpy(float)[f])
        X,Xv,Xt=[sc.transform(df[features].to_numpy(float)[ids]) for ids in [f,v,t]]
        w=np.where(y[f]==1,2.33,1.);wv=np.where(y[v]==1,2.33,1.)
        def bce(z,yy,ww):return float(np.sum(ww*(np.logaddexp(0,z)-yy*z))/ww.sum())
        if row.model=='original':
            model=SimpleANFIS(len(features),learning_rate=.02,epochs=500,seed=seed);model.initialize(X)
            a=model.normalize_strengths(model.calculate_strengths(X));av=model.normalize_strengths(model.calculate_strengths(Xv))
            xb=np.column_stack([X,np.ones(len(f))]);xv=np.column_stack([Xv,np.ones(len(v))]);history=[]
            for epoch in range(501):
                z=(a*(xb@model.consequents.T)).sum(axis=1);zv=(av*(xv@model.consequents.T)).sum(axis=1)
                grad=(a*((expit(z)-y[f])*w)[:,None]).T@xb/w.sum()
                history.append(dict(epoch=epoch,training_bce=bce(z,y[f],w),validation_bce=bce(zv,y[v],wv),gradient_l2=float(np.linalg.norm(grad))))
                if epoch<500:model.consequents-=.02*grad
            best_epoch=500
        else:
            model=ResearchANFIS(len(features),seed,lr=.01,l2=.001).fit(X,y[f],Xv,y[v],w,wv)
            history=model.history;best_epoch=model.best_epoch
        threshold=choose_threshold(y[v],model.predict_proba(Xv));measured=metrics(y[t],model.predict_proba(Xt),threshold)
        for key in measured:
            assert abs(measured[key]-row[key])<1e-10,(row.feature_set,row.model,key,measured[key],row[key])
        assert abs(threshold-row.threshold)<1e-12
        modelcoef=model.consequents if row.model=='original' else model.coef
        if row.model=='original':
            gradnorm=history[-1]['gradient_l2'];initial=history[0]['training_bce']
        else:
            _,grads=model.objective(X,y[f],w,True);gradnorm=float(np.linalg.norm(grads[0]));initial=None
        convergence.append(dict(seed=seed,feature_set=row.feature_set,d=int(row.d),model=row.model,
            epochs_run=int(row.epochs_run),best_epoch=best_epoch,initial_training_bce=initial,
            selected_training_bce=bce(model.forward(X),y[f],w),selected_validation_bce=bce(model.forward(Xv),y[v],wv),
            selected_consequent_gradient_l2=gradnorm,coefficient_l2=float(np.linalg.norm(modelcoef)),
            agreement='all saved dimension metrics and validation threshold agree within 1e-10'))
        stem=f'{seed}_{row.feature_set}_{row.model}'
        pd.DataFrame(history).to_csv(dest/f'{stem}_trajectory.csv',index=False)
        with (dest/f'{stem}.pkl').open('wb') as file:pickle.dump(dict(model=model,scaler=sc,features=features,
            fit_ids=f,val_ids=v,test_ids=t,threshold=threshold,provenance='deterministic convergence replay; not timed benchmark'),file)
    pd.DataFrame(convergence).to_csv(OUT/'dimension_convergence.csv',index=False)
    print('30 convergence replays agreed with completed dimension metrics within 1e-10')


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
