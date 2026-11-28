import argparse
import json
from collections import Counter
from pathlib import Path
import random
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from model_utils import spectrogram_from_audio, features_from_spectrogram
from train_model import split_class, recording_info

DATA = Path(__file__).resolve().parents[1] / 'audio_augmentation' / 'audio_samples'
OUT = Path(__file__).resolve().parent / 'artifacts'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if not a.data.is_dir():
        p.error(f'Data saknas: {a.data}. Ange --data <sökväg till audio_samples>.')
    rng=random.Random(a.seed)
    parts={k:[] for k in ('train','val','test')}
    groups={}
    caveats=[]
    for folder in sorted(x for x in a.data.iterdir() if x.is_dir()):
        wav=sorted(folder.glob('*.wav'))
        if not wav: continue
        split,independent=split_class(wav,rng)
        for k in parts:
            parts[k].extend((f,folder.name) for f in split[k])
        n=len(set(recording_info(f)[0] for f in wav))
        groups[folder.name]={'clips':len(wav),'source_recordings':n}
        if not independent: caveats.append(f'{folder.name}: {n} källinspelning(ar); testet delar ursprung med träningen')
    data={}
    for name,pairs in parts.items():
        print('Bearbetar',name,len(pairs),flush=True)
        t=time.time()
        xx=np.stack([features_from_spectrogram(spectrogram_from_audio(f)) for f,_ in pairs])
        yy=np.array([label for _,label in pairs])
        data[name]=(xx,yy)
        print('Klart efter',round(time.time()-t,1),'sekunder',flush=True)
    models={
      'ExtraTrees':ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,class_weight='balanced',random_state=a.seed,n_jobs=-1),
      'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=a.seed))
    }
    classes=sorted(set(data['train'][1]))
    mapping={'felm4':'FEL','korrm4':'OK','medm8':'OK','utanm8':'FEL','bakgrund':'INAKTIV'}
    results={'seed':a.seed,'groups':groups,'counts':{k:dict(Counter(y)) for k,(_,y) in data.items()},'classes':classes,'limitations':caveats+['Delade sessioner eller närliggande tidsklipp kan överskatta generalisering','Ingen live-Pi-mätning eller ny bakgrundsklass verifierad'],'models':{}}
    for name,model in models.items():
        model.fit(*data['train'])
        evaluations={}
        for split in ('val','test'):
            xx,yy=data[split]; pred=model.predict(xx)
            cm=confusion_matrix(yy,pred,labels=classes)
            actual_status=np.array([mapping.get(z,'OSÄKER') for z in yy])
            pred_status=np.array([mapping.get(z,'OSÄKER') for z in pred])
            missed=int(np.sum((actual_status=='FEL') & (pred_status=='OK')))
            false=int(np.sum((actual_status=='OK') & (pred_status=='FEL')))
            evaluations[split]={'accuracy':float(accuracy_score(yy,pred)), 'balanced_accuracy':float(balanced_accuracy_score(yy,pred)), 'confusion_matrix':cm.tolist(), 'classification_report':classification_report(yy,pred,labels=classes,output_dict=True,zero_division=0), 'binary_status_accuracy':float(np.mean(actual_status==pred_status)), 'missed_faults':missed,'false_alarms':false,'n':len(yy)}
        results['models'][name]={'validation':evaluations['val'],'test':evaluations['test']}
        print(name,'val',round(evaluations['val']['accuracy'],4),'test',round(evaluations['test']['accuracy'],4),'missade fel',evaluations['test']['missed_faults'],'falska larm',evaluations['test']['false_alarms'],flush=True)
        a.output.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,a.output/('model.joblib' if name=='ExtraTrees' else 'logistic_baseline.joblib'),compress=3)
    (a.output/'comparison.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Sparade comparison.json',flush=True)
if __name__=='__main__':main()
