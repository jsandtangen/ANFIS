import json
from pathlib import Path
import pickle
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from main import SimpleANFIS, FEATURES, choose_threshold
from research_models import ResearchANFIS
from research_pipeline import split, balance, metrics


class NumericalTests(unittest.TestCase):
    def test_premise_and_consequent_gradients(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(11,2));y=rng.integers(0,2,11)
        model=ResearchANFIS(2,trainable=True,l2=.01);model.initialize(X)
        w=np.where(y==1,2.33,1.)
        loss,grads=model.objective(X,y,w,True)
        for param,grad in zip([model.coef,model.centers,model.log_widths],grads):
            for index in np.ndindex(param.shape):
                old=param[index];eps=1e-6
                param[index]=old+eps;plus=model.objective(X,y,w)
                param[index]=old-eps;minus=model.objective(X,y,w)
                param[index]=old
                self.assertAlmostEqual(grad[index],(plus-minus)/(2*eps),places=7)

    def test_stable_inference_and_original_agreement(self):
        X=np.random.default_rng(5).normal(size=(20,4))
        original=SimpleANFIS(4);original.initialize(X)
        stable=ResearchANFIS(4);stable.initialize(X)
        stable.coef=original.consequents.copy()
        np.testing.assert_allclose(original.forward(X),stable.forward(X),atol=1e-9)
        _,_,a,_,z=stable.parts(np.full((2,4),1000.))
        np.testing.assert_allclose(a.sum(axis=1),1,atol=1e-9)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((np.exp(stable.log_widths)>0).all())

    def test_paired_split_and_resampling_integrity(self):
        y=np.r_[np.zeros(700,dtype=int),np.ones(300,dtype=int)]
        for seed in [42,7,19,73,101]:
            f,v,t=split(y,seed)
            self.assertEqual([len(f),len(v),len(t)],[640,160,200])
            self.assertFalse(set(f)&set(v) or set(f)&set(t) or set(v)&set(t))
            self.assertEqual(np.bincount(y[t]).tolist(),[140,60])
            X=np.arange(640)[:,None]
            xf,yf,w,idx=balance(X,y[f],'oversampled',seed)
            self.assertEqual(np.bincount(yf).tolist(),[448,448])
            np.testing.assert_array_equal(xf,X[idx])

    def test_metrics_and_threshold(self):
        m=metrics(np.array([0,0,1,1]),np.array([.1,.6,.4,.8]),.5)
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertEqual(m['balanced_accuracy'],.5)
        t=choose_threshold([0,1],np.array([.1,.9]))
        self.assertAlmostEqual(t,.2)


class ArtifactTests(unittest.TestCase):
    @unittest.skipUnless(Path('results/research/cases/case_summaries.json').exists(),'run pipeline first')
    def test_case_reconstruction_against_checkpoint(self):
        out=Path('results/research');bundle=pickle.loads((out/'models/42_weighted_fixed_adam.pkl').read_bytes())
        model=bundle['model'];df=pd.read_csv('data/german_credit_data.csv')
        summaries=json.loads((out/'cases/case_summaries.json').read_text())
        self.assertGreaterEqual(len(summaries),2)
        correct=[]
        for case in summaries:
            row=case['row_zero_based'];self.assertIn(row,bundle['test_ids'])
            X=bundle['scaler'].transform(df[FEATURES].to_numpy(float)[row:row+1])
            table=pd.read_csv(out/case['rules_csv'])
            _,_,a,f,z=model.parts(X)
            np.testing.assert_allclose(table.normalized_firing,a[0],atol=1e-14)
            np.testing.assert_allclose(table.consequent,f[0],atol=1e-14)
            self.assertAlmostEqual(table.contribution.sum(),z[0],places=12)
            self.assertAlmostEqual(expit(z[0]),case['score'],places=12)
            self.assertAlmostEqual(case['reconstructed_logit'],case['direct_logit'],places=12)
            correct.append(case['true_label']==case['predicted_label'])
        self.assertEqual(sorted(correct),[False,True])

    @unittest.skipUnless(Path('results/research/per_split_metrics.csv').exists(),'run pipeline first')
    def test_saved_metrics_from_predictions(self):
        out=Path('results/research');r=pd.read_csv(out/'per_split_metrics.csv');p=pd.read_csv(out/'test_predictions.csv')
        self.assertEqual(len(r),5*3*6*2)
        for _,row in r.iterrows():
            sub=p[(p.seed==row.seed)&(p.condition==row.condition)&(p.model==row.model)]
            self.assertEqual(len(sub),200)
            m=metrics(sub.label.to_numpy(),sub.score.to_numpy(),row.threshold)
            for key,value in m.items(): self.assertAlmostEqual(value,row[key],places=12)


if __name__=='__main__':unittest.main(verbosity=2)
