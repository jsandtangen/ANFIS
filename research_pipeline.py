"""Bounded, matched-split extension of the preserved ACIT4620 experiment.

Run after reproduce_original.py. Test sets never enter fit or trial selection.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import pickle
import platform
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve)
from threadpoolctl import threadpool_limits
from main import SimpleANFIS, FEATURES, choose_threshold
from research_models import ResearchANFIS

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/research'
CONFIG = dict(seeds=[42,7,19,73,101], conditions=['none','weighted','oversampled'],
    fit_size=640, validation_size=160, test_size=200,
    criterion='maximum validation balanced accuracy; first trial wins exact ties',
    threshold='0.20 to 0.80 inclusive by .01; original choose_threshold first maximum',
    anfis_trials=[dict(lr=.01,l2=.001),dict(lr=.03,l2=.01)],
    mlp_trials=[dict(lr=.001,alpha=.0001),dict(lr=.003,alpha=.01)],
    logistic_C=[.1,1.], max_epochs=400, patience=50,
    feature_names=FEATURES, dimension_seeds=[42,7,19],
    case_criteria='seed 42 weighted tuned fixed ANFIS: lowest row id correct bad; lowest row id incorrect good; fallback lowest id in requested correctness group',
    status='predeclared before execution of new experiments; exploratory extension of previously inspected project')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist()), encoding='utf-8')


def metrics(y, p, t):
    pred = p >= t
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0),specificity=tn/(tn+fp),roc_auc=roc_auc_score(y,p),
        average_precision=average_precision_score(y,p), brier=brier_score_loss(y,p),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp), positive_rate=float(pred.mean()))


def split(y, seed):
    train,test = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
    fit,val = train_test_split(train,test_size=.2,stratify=y[train],random_state=seed)
    return fit,val,test


def balance(X,y,condition,seed):
    idx = np.arange(len(y))
    weights = np.ones(len(y))
    if condition == 'weighted':
        weights = np.where(y==1, 2.33, 1.)
    elif condition == 'oversampled':
        rng=np.random.default_rng(seed)
        good,bad=np.where(y==0)[0],np.where(y==1)[0]
        idx=np.concatenate([good,bad,rng.choice(bad,len(good)-len(bad),replace=True)])
        rng.shuffle(idx)
        weights=np.ones(len(idx))
    return X[idx],y[idx],weights,idx


def mlp_fit(X,y,Xv,yv,w,seed,lr,alpha):
    # External validation avoids duplicate leakage into sklearn internal early stopping.
    model=MLPClassifier(hidden_layer_sizes=(16,8),learning_rate_init=lr,alpha=alpha,
                        random_state=seed,early_stopping=False,max_iter=1)
    best=-np.inf; wait=0; history=[]; saved=None
    for epoch in range(1,401):
        model.partial_fit(X,y,classes=[0,1],sample_weight=w)
        pv=model.predict_proba(Xv)[:,1]
        t=choose_threshold(yv,pv)
        score=balanced_accuracy_score(yv,pv>=t)
        history.append(dict(epoch=epoch,train_objective=model.loss_,val_balanced_accuracy=score))
        if score > best+1e-6:
            best,wait,saved=score,0,pickle.dumps(model)
        else:
            wait+=1
        if wait>=50:
            break
    return pickle.loads(saved),history


def predict(model,X):
    p=model.predict_proba(X)
    return p[:,1] if p.ndim==2 else p


def diagnostics(model,X,y,tag):
    if isinstance(model,SimpleANFIS):
        raw=model.calculate_strengths(X); a=model.normalize_strengths(raw)
        f=model.rule_outputs(X); z=model.forward(X)
        centers=np.array(model.membership_parameters)[:,:,0]
        widths=np.array(model.membership_parameters)[:,:,1]
        # Same consequents and premises; normalization only changes.
        log_mu=-.5*((X[:,:,None]-centers[None])/widths[None])**2
        from scipy.special import logsumexp
        lw=sum(log_mu[:,j,np.array(model.rules)[:,j]] for j in range(X.shape[1]))
        stable=np.exp(lw-logsumexp(lw,axis=1,keepdims=True))
        coef=model.consequents
        epsilon_delta=float(np.max(np.abs(z-(stable*f).sum(axis=1))))
    else:
        log_mu,lw,a,f,z=model.parts(X); raw=np.exp(lw)
        centers=model.centers; widths=np.exp(model.log_widths);coef=model.coef
        epsilon_delta=None
    mu=np.exp(-.5*((X[:,:,None]-centers[None])/widths[None])**2)
    p=expit(z)
    effective=1/np.sum(a*a,axis=1)
    return dict(tag=tag, rules=int(a.shape[1]),consequent_parameters=int(coef.size),
        trainable_parameters=int(coef.size+(centers.size*2 if getattr(model,'trainable',False) else 0)),
        dense_rule_workspace_bytes=int(a.nbytes+raw.nbytes+f.nbytes),
        memory_note='three N by R float64 arrays only; excludes temporaries, BLAS, interpreter and model; not peak RSS',
        normalized_sum_min=float(a.sum(axis=1).min()),raw_sum_min=float(raw.sum(axis=1).min()),
        rows_raw_sum_below_1e10=int((raw.sum(axis=1)<1e-10).sum()),
        max_epsilon_vs_stable_logit_delta=epsilon_delta,
        mean_max_rule_weight=float(a.max(axis=1).mean()),mean_effective_rules=float(effective.mean()),
        rules_mean_weight_above_001=int((a.mean(axis=0)>.001).sum()),
        low_feature_coverage_fraction=float((mu.max(axis=2)<.01).mean()),
        saturated_membership_fraction=float((mu<1e-8).mean()),
        coincident_center_pairs=int(sum(np.isclose(centers[:,i],centers[:,j]).sum() for i,j in [(0,1),(0,2),(1,2)])),
        coefficient_l2=float(np.linalg.norm(coef)),coefficient_abs_max=float(np.max(np.abs(coef))),
        positive_rate_default=float((p>=.5).mean()),score_min=float(p.min()),score_max=float(p.max()),
        good_score_mean=float(p[y==0].mean()),bad_score_mean=float(p[y==1].mean()),
        centers=centers.tolist(),widths=widths.tolist(),mean_rule_weights=a.mean(axis=0).tolist())


def cases(model,scaler,Xraw,y,ids,threshold):
    X=scaler.transform(Xraw)
    p=predict(model,X); correct=(p>=threshold)==y
    chosen=[]
    for ok,label in [(True,1),(False,0)]:
        candidates=np.where((correct==ok)&(y==label))[0]
        if not len(candidates): candidates=np.where(correct==ok)[0]
        if len(candidates): chosen.append(int(candidates[np.argmin(ids[candidates])]))
    summaries=[]
    for i in chosen:
        row=X[i:i+1]
        logmu,lw,a,f,z=model.parts(row)
        contributions=a[0]*f[0]
        record=int(ids[i]); stem=f'case_row_{record}'
        table=pd.DataFrame(dict(rule_id=np.arange(len(model.rules)),
            antecedent=[' AND '.join(f'{FEATURES[j]}={SimpleANFIS.MF_NAMES[k]}' for j,k in enumerate(rule)) for rule in model.rules],
            firing=np.exp(lw[0]),normalized_firing=a[0],consequent=f[0],contribution=contributions))
        table.to_csv(OUT/'cases'/f'{stem}_rules.csv',index=False)
        membership=pd.DataFrame([dict(feature=FEATURES[j],set=SimpleANFIS.MF_NAMES[k],
            original_value=float(Xraw[i,j]),scaled_value=float(row[0,j]),center=float(model.centers[j,k]),
            width=float(np.exp(model.log_widths[j,k])),grade=float(np.exp(logmu[0,j,k])))
            for j in range(len(FEATURES)) for k in range(3)])
        membership.to_csv(OUT/'cases'/f'{stem}_memberships.csv',index=False)
        direct=float(model.forward(row)[0]); reconstructed=float(contributions.sum())
        item=dict(row_zero_based=record,csv_line_including_header=record+2,true_label=int(y[i]),
            predicted_label=int(p[i]>=threshold),score=float(p[i]),threshold=threshold,
            scaled_inputs=row[0].tolist(),original_inputs=Xraw[i].tolist(),
            direct_logit=direct,reconstructed_logit=reconstructed,error=abs(direct-reconstructed),
            positive_contribution_sum=float(contributions[contributions>0].sum()),
            negative_contribution_sum=float(contributions[contributions<0].sum()),
            top_absolute_rule_ids=np.argsort(-np.abs(contributions))[:5].tolist(),
            rules_csv=f'cases/{stem}_rules.csv',membership_csv=f'cases/{stem}_memberships.csv')
        summaries.append(item)
        top=table.iloc[np.argsort(-np.abs(contributions))[:10]].sort_values('contribution')
        fig,ax=plt.subplots(figsize=(8,4));ax.barh(top.rule_id.astype(str),top.contribution,
            color=['tab:red' if c<0 else 'tab:blue' for c in top.contribution])
        ax.set(xlabel='Contribution to logit (not probability)',ylabel='Rule ID (zero based)',title=f'Held-out row {record}')
        fig.tight_layout();fig.savefig(OUT/'plots'/f'{stem}.png',dpi=160);plt.close(fig)
    dump(OUT/'cases/case_summaries.json',summaries)


def plot_diagnostics(models,X,y):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for name,model in models.items():
        p=predict(model,X)
        fpr,tpr,_=roc_curve(y,p);axes[0,0].plot(fpr,tpr,label=name)
        pr,re,_=precision_recall_curve(y,p);axes[0,1].plot(re,pr,label=name)
        ts=np.linspace(.2,.8,121)
        axes[0,2].plot(ts,[balanced_accuracy_score(y,p>=t) for t in ts],label=name)
        if name=='original':
            for label in [0,1]:axes[1,0].hist(p[y==label],bins=20,alpha=.5,label=f'class {label}')
            axes[1,1].plot(model.training_history,label='original training BCE')
        else:
            h=pd.DataFrame(model.history)
            axes[1,1].plot(h.epoch,h.val_objective,label=name+' validation objective')
            axes[1,2].semilogy(h.epoch,h.gradient_norm,label=name)
    for ax,title in zip(axes.flat,['Test ROC','Test precision-recall','Test threshold sensitivity (diagnostic only)',
                                  'Original test score distribution','Training / validation objectives','Full gradient norm']):
        ax.set_title(title);ax.legend(fontsize=7)
    axes[0,0].set(xlabel='False-positive rate',ylabel='Recall')
    axes[0,1].set(xlabel='Recall',ylabel='Precision')
    axes[0,2].set(xlabel='Threshold',ylabel='Balanced accuracy')
    fig.tight_layout();fig.savefig(OUT/'plots/diagnostics_seed42_weighted.png',dpi=160);plt.close(fig)


def main():
    for folder in ['models','histories','cases','plots']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    dump(OUT/'config.json',CONFIG)
    df=pd.read_csv(ROOT/'data/german_credit_data.csv');y=df.target.map({'good':0,'bad':1}).to_numpy()
    assert len(df)==1000 and np.bincount(y).tolist()==[700,300]
    assert not df.isna().any().any()
    dump(OUT/'data_audit.json',dict(shape=df.shape,columns=df.columns.tolist(),missing=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),target_counts=np.bincount(y),
        numeric_unique={c:sorted(df[c].unique().tolist()) for c in ['payment_to_income_ratio','residence_since','n_credits','n_guarantors']},
        sha256=hashlib.sha256((ROOT/'data/german_credit_data.csv').read_bytes()).hexdigest()))
    records=[]; trials=[]; predictions=[]; splits=[]; diags=[]; timings=[]
    start=time.perf_counter()
    for seed in CONFIG['seeds']:
        fit,val,test=split(y,seed)
        splits.extend(dict(seed=seed,row_id=int(i),partition=part,label=int(y[i]))
                      for part,idx in [('fit',fit),('validation',val),('test',test)] for i in idx)
        raw=df[FEATURES].to_numpy(float)
        scaler=StandardScaler().fit(raw[fit]);X,Xv,Xt=[scaler.transform(raw[i]) for i in [fit,val,test]]
        for condition in CONFIG['conditions']:
            print(f'seed={seed} condition={condition}',flush=True)
            Xf,yf,w,idx=balance(X,y[fit],condition,seed)
            vweights=np.where(y[val]==1,2.33,1.) if condition!='none' else np.ones(len(val))
            choices={}; search_times={}
            for name in ['original','fixed_adam','trainable_adam','mlp','logistic','dummy']:
                t0=time.perf_counter(); candidates=[]
                configs=CONFIG['anfis_trials'] if name in ['fixed_adam','trainable_adam'] else (
                    CONFIG['mlp_trials'] if name=='mlp' else ([dict(C=c) for c in CONFIG['logistic_C']] if name=='logistic' else [{}]))
                for trial,cfg in enumerate(configs):
                    trial_start=time.perf_counter()
                    history=[]
                    if name=='original':
                        model=SimpleANFIS(len(FEATURES),learning_rate=.02,epochs=500,seed=seed)
                        with contextlib.redirect_stdout(None):
                            model.fit(Xf,yf,{0:1.,1:2.33} if condition=='weighted' else None)
                        history=[dict(epoch=i+1,train_objective=float(loss)) for i,loss in enumerate(model.training_history)]
                    elif name in ['fixed_adam','trainable_adam']:
                        model=ResearchANFIS(4,seed,trainable=name=='trainable_adam',**cfg)
                        model.fit(Xf,yf,Xv,y[val],w,vweights);history=model.history
                    elif name=='mlp':
                        model,history=mlp_fit(Xf,yf,Xv,y[val],w,seed,**cfg)
                    elif name=='logistic':
                        model=LogisticRegression(C=cfg['C'],max_iter=1000,random_state=seed).fit(Xf,yf,sample_weight=w)
                    else:
                        model=None
                    pv=np.repeat(np.mean(y[fit]),len(val)) if model is None else predict(model,Xv)
                    threshold=choose_threshold(y[val],pv)
                    score=balanced_accuracy_score(y[val],pv>=threshold)
                    elapsed=time.perf_counter()-trial_start
                    trialrow=dict(seed=seed,condition=condition,model=name,trial=trial,config=json.dumps(cfg),
                        validation_balanced_accuracy=score,threshold=threshold,fit_seconds=elapsed,
                        epochs_run=len(history),best_epoch=getattr(model,'best_epoch',None))
                    trials.append(trialrow)
                    if history:pd.DataFrame(history).to_csv(OUT/'histories'/f'{seed}_{condition}_{name}_{trial}.csv',index=False)
                    candidates.append((score,model,threshold,trialrow))
                # Stable maximum uses candidate order as predeclared tie breaker.
                best=max(candidates,key=lambda item:item[0]);choices[name]=best
                search_times[name]=time.perf_counter()-t0
                score,model,threshold,trialrow=best
                pt=np.repeat(np.mean(y[fit]),len(test)) if model is None else predict(model,Xt)
                modelpath=OUT/'models'/f'{seed}_{condition}_{name}.pkl'
                with modelpath.open('wb') as f:pickle.dump(dict(model=model,scaler=scaler,features=FEATURES,
                    seed=seed,threshold=threshold,fit_ids=fit,val_ids=val,test_ids=test,config=trialrow),f)
                for mode,t in [('default',.5),('validation',threshold)]:
                    records.append(dict(seed=seed,condition=condition,model=name,threshold_mode=mode,threshold=t,
                        selected_trial=trialrow['trial'],selected_fit_seconds=trialrow['fit_seconds'],
                        search_seconds=search_times[name],**metrics(y[test],pt,t)))
                predictions.extend(dict(seed=seed,condition=condition,model=name,row_id=int(row),label=int(label),score=float(p),threshold=threshold)
                                   for row,label,p in zip(test,y[test],pt))
                if name in ['original','fixed_adam','trainable_adam']:
                    diags.append(dict(seed=seed,condition=condition,**diagnostics(model,Xt,y[test],name)))
            if seed==42 and condition=='weighted':
                cases(choices['fixed_adam'][1],scaler,raw[test],y[test],test,choices['fixed_adam'][2])
                plot_diagnostics({name:choices[name][1] for name in ['original','fixed_adam','trainable_adam']},Xt,y[test])
    pd.DataFrame(records).to_csv(OUT/'per_split_metrics.csv',index=False)
    pd.DataFrame(trials).to_csv(OUT/'all_trials.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(OUT/'split_assignments.csv',index=False)
    dump(OUT/'diagnostics.json',diags)
    values=['accuracy','balanced_accuracy','precision','recall','f1','specificity','roc_auc','average_precision','brier','positive_rate','selected_fit_seconds','search_seconds']
    result=pd.DataFrame(records)
    result.groupby(['condition','model','threshold_mode'])[values].agg(['mean','std','median','min','max']).to_csv(OUT/'aggregate_metrics.csv')
    differences=[]
    for condition in CONFIG['conditions']:
        for mode in ['default','validation']:
            sub=result[(result.condition==condition)&(result.threshold_mode==mode)]
            for a,b in [('fixed_adam','mlp'),('trainable_adam','fixed_adam'),('fixed_adam','original'),('fixed_adam','logistic')]:
                paired=sub[sub.model==a].set_index('seed')[values[:9]]-sub[sub.model==b].set_index('seed')[values[:9]]
                for seed,row in paired.iterrows():differences.append(dict(seed=seed,condition=condition,threshold_mode=mode,comparison=f'{a} minus {b}',**row.to_dict()))
    diff=pd.DataFrame(differences);diff.to_csv(OUT/'paired_differences.csv',index=False)
    diff.groupby(['condition','threshold_mode','comparison'])[values[:9]].agg(['mean','std','median','min','max']).to_csv(OUT/'paired_summary.csv')
    dimension_experiments(df,y)
    # Initialization sensitivity with exactly the same data and no model reselection.
    bundle=pickle.loads((OUT/'models/42_weighted_original.pkl').read_bytes())
    f,v,t=bundle['fit_ids'],bundle['val_ids'],bundle['test_ids'];sc=bundle['scaler']
    X=sc.transform(df[FEATURES].to_numpy(float)); init_rows=[]
    for init in [0,1,2,3,4]:
        model=SimpleANFIS(4,learning_rate=.02,epochs=500,seed=init)
        with contextlib.redirect_stdout(None):model.fit(X[f],y[f],{0:1.,1:2.33})
        th=choose_threshold(y[v],model.predict_proba(X[v]));pt=model.predict_proba(X[t])
        for mode,threshold in [('default',.5),('validation',th)]:
            init_rows.append(dict(initialization_seed=init,threshold_mode=mode,threshold=threshold,**metrics(y[t],pt,threshold)))
    pd.DataFrame(init_rows).to_csv(OUT/'initialization_sensitivity.csv',index=False)
    dump(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        processor=platform.processor(),logical_cpus=os.cpu_count(),threads=1,
        wall_seconds=time.perf_counter()-start,packages={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','threadpoolctl']},
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['main.py','research_models.py','research_pipeline.py','reproduce_original.py']},
        memory='CIM RAM query denied; dimension workspace is an explicit array estimate, not measured process peak'))
    print('Completed new experiments',flush=True)


def dimension_experiments(df,y):
    rows=[];ranking=[]
    extras=['residence_since','n_credits','n_guarantors']
    for seed in CONFIG['dimension_seeds']:
        fit,val,test=split(y,seed)
        mi=mutual_info_classif(df[extras].to_numpy()[fit],y[fit],discrete_features=True,random_state=seed)
        order=sorted(range(len(extras)),key=lambda j:(-mi[j],j))
        ranking.extend(dict(seed=seed,feature=extras[j],mutual_information=float(mi[j]),rank=i+1) for i,j in enumerate(order))
        sets=[('base',FEATURES),('original_additions_5',FEATURES+extras[:1]),
              ('original_additions_6',FEATURES+extras[:2]),
              ('train_MI_5',FEATURES+[extras[order[0]]]),
              ('train_MI_6',FEATURES+[extras[j] for j in order[:2]])]
        for label,features in sets:
            raw=df[features].to_numpy(float);sc=StandardScaler().fit(raw[fit])
            X,Xv,Xt=[sc.transform(raw[i]) for i in [fit,val,test]]
            for name in ['original','fixed_adam']:
                # Fixed protocol here, no dimension-specific search; limits scope and compute.
                start=time.perf_counter()
                if name=='original':
                    model=SimpleANFIS(len(features),learning_rate=.02,epochs=500,seed=seed)
                    with contextlib.redirect_stdout(None):model.fit(X,y[fit],{0:1.,1:2.33})
                else:
                    model=ResearchANFIS(len(features),seed,lr=.01,l2=.001)
                    model.fit(X,y[fit],Xv,y[val],np.where(y[fit]==1,2.33,1.),np.where(y[val]==1,2.33,1.))
                elapsed=time.perf_counter()-start
                th=choose_threshold(y[val],predict(model,Xv))
                diag=diagnostics(model,Xt,y[test],name)
                rows.append(dict(seed=seed,feature_set=label,features=';'.join(features),d=len(features),model=name,
                    fit_seconds=elapsed,epochs_run=len(model.training_history if name=='original' else model.history),
                    rules=diag['rules'],consequent_parameters=diag['consequent_parameters'],
                    dense_rule_workspace_bytes=diag['dense_rule_workspace_bytes'],
                    mean_effective_rules=diag['mean_effective_rules'],normalized_sum_min=diag['normalized_sum_min'],
                    threshold=th,**metrics(y[test],predict(model,Xt),th)))
                # Persist each completed fit so an interrupted dimension run is reviewable.
                pd.DataFrame(rows).to_csv(OUT/'dimension_metrics.csv',index=False)
                pd.DataFrame(ranking).to_csv(OUT/'train_only_feature_ranking.csv',index=False)
                print('dimension',seed,label,name,round(elapsed,2),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'dimension_metrics.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'train_only_feature_ranking.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
