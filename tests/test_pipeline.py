"""Checks for indexing, leakage prevention, and the actual saved artifacts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from scipy.io import loadmat
from emg_assignment import ROOT, equal_slices, trial_features, standardize_train_test, fold_ids, evaluate


class PipelineTests(unittest.TestCase):
    def test_all_samples_used_once(self):
        for length in (9205,10000,15106):
            seen=np.zeros(length,int)
            for trial in equal_slices(length,15):
                local=np.zeros(trial.stop-trial.start,int)
                for segment in equal_slices(len(local),3):
                    local[segment]+=1
                np.testing.assert_array_equal(local,1)
                seen[trial]+=1
            np.testing.assert_array_equal(seen,1)

    def test_known_features_and_channel_order(self):
        signal=np.tile(np.arange(90,dtype=float)[:,None],(1,8))+100*np.arange(8)
        result=trial_features(signal,15,3)
        self.assertEqual(result.shape,(15,48))
        np.testing.assert_allclose(result[0,:6],[.5,.5,2.5,.5,4.5,.5])
        np.testing.assert_allclose(result[0,6:12],[100.5,.5,102.5,.5,104.5,.5])
        self.assertAlmostEqual(result[-1,-2],788.5)

    def test_test_outlier_does_not_influence_scaler(self):
        train=np.array([[0.,7.],[2.,7.],[4.,7.]])
        tr,te=standardize_train_test(train,np.array([[1000.,7.]]))
        np.testing.assert_allclose(tr[:,0],[-1,0,1])
        np.testing.assert_allclose(tr[:,1],0)
        np.testing.assert_allclose(te,[[499,0]])

    def test_stratification_and_knn(self):
        y=np.repeat([1,2],60)
        folds=fold_ids(y,10,2026)
        for fold in range(1,11):
            np.testing.assert_array_equal(np.bincount(y[folds==fold])[1:],[6,6])
        X=np.column_stack([y.astype(float),np.ones(len(y))])
        accuracy,pred=evaluate(X,y,folds,5)
        self.assertEqual(accuracy,1)
        np.testing.assert_array_equal(pred,y)

    def test_saved_dataset_and_plan(self):
        import pandas as pd
        data=loadmat(ROOT/'artifacts/emg_features.mat')
        self.assertEqual(data['X'].shape,(1200,48))
        self.assertEqual(data['data'].shape,(1200,49))
        np.testing.assert_array_equal(data['data'][:,-1],data['y'].ravel())
        self.assertTrue(np.isfinite(data['X']).all())
        self.assertTrue((data['X'][:,1::2]>=0).all())
        manifest=pd.read_csv(ROOT/'artifacts/dataset_manifest.csv')
        self.assertEqual(len(manifest),80)
        self.assertEqual(manifest.sha256.nunique(),80)
        self.assertFalse(manifest.path.str.contains('sub3/sub3').any())
        plan=loadmat(ROOT/'artifacts/experiment_plan.mat')
        y=data['y'].ravel()
        for mi,m in enumerate(range(2,20)):
            for ci in range(10):
                labels=plan['selected_labels'][mi,ci,:m]
                self.assertEqual(len(set(labels)),m)
                mask=np.isin(y,labels)
                self.assertEqual(mask.sum(),60*m)
                self.assertTrue((plan['fold_ids'][mi,ci,:,~mask]==0).all())
                for mc in range(10):
                    folds=plan['fold_ids'][mi,ci,mc,mask]
                    self.assertTrue(np.all(np.bincount(folds)[1:]==6*m))


if __name__=='__main__':
    unittest.main()
