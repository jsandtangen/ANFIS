"""Replay original GD with cached fixed premises to expose convergence diagnostics.

The cached vectorized update is tested against the unchanged saved model.
"""
import contextlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp
from main import SimpleANFIS, FEATURES
from threadpoolctl import threadpool_limits

OUT=Path('results/research')


def run():
    bundle=pickle.loads((OUT/'models/42_weighted_original.pkl').read_bytes())
    df=pd.read_csv('data/german_credit_data.csv');y=df.target.map({'good':0,'bad':1}).to_numpy()
    scaler=bundle['scaler'];X=scaler.transform(df[FEATURES].to_numpy(float))
    fit,val,test=[bundle[k] for k in ['fit_ids','val_ids','test_ids']]
    original=bundle['model'];model=SimpleANFIS(4,learning_rate=.02,epochs=500,seed=42);model.initialize(X[fit])
    a=model.normalize_strengths(model.calculate_strengths(X[fit]));av=model.normalize_strengths(model.calculate_strengths(X[val]))
    xb=np.column_stack([X[fit],np.ones(len(fit))]);xv=np.column_stack([X[val],np.ones(len(val))])
    w=np.where(y[fit]==1,2.33,1.);wv=np.where(y[val]==1,2.33,1.)
    def loss(z,yy,ww):return float(np.sum(ww*(np.logaddexp(0,z)-yy*z))/ww.sum())
    h=[]
    for epoch in range(501):
        z=(a*(xb@model.consequents.T)).sum(axis=1)
        zv=(av*(xv@model.consequents.T)).sum(axis=1)
        grad=(a*((expit(z)-y[fit])*w)[:,None]).T@xb/w.sum()
        h.append(dict(epoch=epoch,training_bce=loss(z,y[fit],w),validation_bce=loss(zv,y[val],wv),
            gradient_l2=float(np.linalg.norm(grad)),parameter_l2=float(np.linalg.norm(model.consequents)),
            max_rule_gradient=float(np.linalg.norm(grad,axis=1).max())))
        if epoch<500:model.consequents-=.02*grad
    error=float(np.max(np.abs(model.consequents-original.consequents)))
    assert error<1e-12,error
    pd.DataFrame(h).to_csv(OUT/'original_gradient_trajectory.csv',index=False)
    # Same initialization, same fixed weights, same 500 full-batch steps, LR only.
    ablations=[]
    for lr in [.02,.2,2.]:
        m=SimpleANFIS(4,learning_rate=lr,epochs=500,seed=42)
        with contextlib.redirect_stdout(None):m.fit(X[fit],y[fit],{0:1.,1:2.33})
        zv=m.forward(X[val]);p=expit(zv)
        from main import choose_threshold
        from research_pipeline import metrics
        threshold=choose_threshold(y[val],p)
        ablations.append(dict(lr=lr,train_loss_last_recorded=m.training_history[-1],
            validation_bce=loss(zv,y[val],wv),validation_threshold=threshold,
            **{'validation_'+k:v for k,v in metrics(y[val],p,threshold).items()}))
    pd.DataFrame(ablations).to_csv(OUT/'learning_rate_validation_ablation.csv',index=False)
    # This ablation is diagnostic only; no test-based hyperparameter selection.
    diag=json.loads((OUT/'diagnostics.json').read_text())
    d=next(d for d in diag if d['seed']==42 and d['condition']=='weighted' and d['tag']=='original')
    (OUT/'telemetry_verification.json').write_text(json.dumps(dict(max_coefficient_difference=error,
        initial=h[0],final=h[-1],normalization_only_max_test_logit_difference=d['max_epsilon_vs_stable_logit_delta']),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4));h=pd.DataFrame(h)
    axes[0].plot(h.epoch,h.training_bce,label='fit');axes[0].plot(h.epoch,h.validation_bce,label='validation');axes[0].legend()
    axes[0].set(xlabel='Epoch',ylabel='Weighted BCE',title='Original GD replay')
    axes[1].plot(h.epoch,h.gradient_l2);axes[1].set(xlabel='Epoch',ylabel='Gradient L2 norm')
    params=np.array(original.membership_parameters)
    for j,feature in enumerate(FEATURES):
        # Coverage plot in scaled coordinates; colors denote feature and line style set.
        grid=np.linspace(-3,5,300)
        for k in range(3):axes[2].plot(grid,np.exp(-.5*((grid-params[j,k,0])/params[j,k,1])**2),
            label=f'{feature} {k}',alpha=.7,linestyle=['-',':','--'][k])
    axes[2].set(xlabel='Scaled input',ylabel='Membership grade');axes[2].legend(fontsize=5)
    fig.tight_layout();fig.savefig(OUT/'plots/original_optimization_and_memberships.png',dpi=160);plt.close(fig)
    # Mean rule usage; both classes are descriptive test analysis.
    fig,ax=plt.subplots(figsize=(10,4))
    for name in ['original','fixed_adam','trainable_adam']:
        item=next(d for d in diag if d['seed']==42 and d['condition']=='weighted' and d['tag']==name)
        ax.plot(sorted(item['mean_rule_weights'],reverse=True),label=name)
    ax.set(xlabel='Rule rank by test mean weight',ylabel='Mean normalized firing',title='Rule concentration, seed 42 weighted')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/rule_concentration.png',dpi=160);plt.close(fig)
    # Conditional test-sample uncertainty; never pool overlapping repeated test sets.
    pred=pd.read_csv(OUT/'test_predictions.csv')
    a=pred[(pred.seed==42)&(pred.condition=='weighted')&(pred.model=='fixed_adam')].set_index('row_id').sort_index()
    b=pred[(pred.seed==42)&(pred.condition=='weighted')&(pred.model=='mlp')].set_index('row_id').sort_index()
    np.testing.assert_array_equal(a.index,b.index)
    labels=a.label.to_numpy();rng=np.random.default_rng(20261112);samples=[]
    for repetition in range(2000):
        ix=np.concatenate([rng.choice(np.where(labels==c)[0],int((labels==c).sum()),replace=True) for c in [0,1]])
        ma=metrics(labels[ix],a.score.to_numpy()[ix],float(a.threshold.iloc[0]))
        mb=metrics(labels[ix],b.score.to_numpy()[ix],float(b.threshold.iloc[0]))
        samples.append({key:ma[key]-mb[key] for key in ['balanced_accuracy','recall','f1','roc_auc','average_precision']})
    bootstrap=pd.DataFrame(samples);bootstrap.to_csv(OUT/'conditional_bootstrap_samples.csv',index=False)
    pd.DataFrame([dict(metric=k,lower=float(bootstrap[k].quantile(.025)),upper=float(bootstrap[k].quantile(.975)),
        repetitions=2000,seed=20261112,comparison='fixed_adam minus mlp; seed42 weighted validation thresholds',
        interpretation='stratified paired percentile interval conditional on fitted models and test partition; no training/tuning uncertainty')
        for k in bootstrap]).to_csv(OUT/'conditional_bootstrap.csv',index=False)
    print('Telemetry replay coefficient error:',error)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
