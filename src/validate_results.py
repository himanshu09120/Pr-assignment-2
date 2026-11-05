"""Independent consistency checks for final metrics, stored predictions, and PDF."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pypdf import PdfReader
from emg_assignment import ROOT, standardize_train_test


def main():
    r=ROOT/'artifacts/results'
    summary=json.loads((r/'summary.json').read_text())
    runs=pd.read_csv(r/'accuracy_runs.csv')
    assert len(runs)==1800
    assert runs.accuracy.between(0,1).all()
    mat=loadmat(r/'classification_results.mat')
    assert mat['accuracy'].shape==(18,10,10)
    np.testing.assert_allclose(mat['accuracy'].ravel(),runs.accuracy.to_numpy(),rtol=0,atol=1e-15)
    curve=pd.read_csv(r/'accuracy_curve.csv')
    np.testing.assert_allclose(curve.mean_accuracy,mat['accuracy'].mean(axis=(1,2)),atol=1e-15)
    pred=pd.read_csv(r/'out_of_fold_predictions.csv')
    for kind,n,labels in [('binary',120,[9,17]),('full',1200,list(range(1,21)))]:
        subset=pred[pred.experiment==kind]
        assert len(subset)==n*10
        assert subset.groupby('mc').row.nunique().eq(n).all()
        measured=subset.groupby('mc').apply(lambda x: (x.actual==x.predicted).mean(),include_groups=False)
        np.testing.assert_allclose(measured,summary[kind+'_scores'])
        confusion=pd.crosstab(subset.actual,subset.predicted).reindex(index=labels,columns=labels,fill_value=0)
        np.testing.assert_array_equal(confusion,np.loadtxt(r/(kind+'_confusion.csv'),delimiter=','))
    # Independently recompute one complete 20-class fold by direct distances
    # and majority counts, avoiding sklearn's distance/voting implementation.
    data=loadmat(ROOT/'artifacts/emg_features.mat')
    plan=loadmat(ROOT/'artifacts/experiment_plan.mat')
    X,y=data['X'],data['y'].ravel()
    folds=plan['full_fold_ids'][0]
    test=folds==1
    train_X,test_X=standardize_train_test(X[~test],X[test])
    expected=[]
    for row in test_X:
        d=((train_X-row)**2).sum(axis=1)
        nearest=np.argsort(d,kind='stable')[:5]
        expected.append(np.argmax(np.bincount(y[~test][nearest],minlength=21)))
    actual=pred[(pred.experiment=='full')&(pred.mc==1)&(pred.fold==1)].sort_values('row')
    np.testing.assert_array_equal(expected,actual.predicted)
    pdf=ROOT/'report/EMG_Classification_Report.pdf'
    reader=PdfReader(pdf)
    assert 3<=len(reader.pages)<=10, f'Report must be 3--10 pages, got {len(reader.pages)}'
    text='\n'.join(page.extract_text() for page in reader.pages)
    assert '@@' not in text and '??' not in text
    for label in ('binary_mean','full_mean','loso_mean'):
        assert f'{100*summary[label]:.2f}' in text
    result=dict(status='PASS',report_pages=len(reader.pages),cv_results=1800,
                oof_predictions=len(pred),independently_checked_predictions=int(test.sum()),
                checks=['accuracy tensor/CSV equality','recomputed summary metrics',
                        'confusion matrices from predictions','independent kNN fold',
                        'PDF page count, metric values, and resolved references'])
    (ROOT/'artifacts/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
