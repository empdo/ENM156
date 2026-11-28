import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

from model_utils import spectrogram_from_audio, features_from_spectrogram

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'audio_augmentation' / 'audio_samples'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'artifacts'
PATTERN = re.compile(r'_([0-9a-f]{8})_(\d+)(?:_padded)?$')


def recording_info(path):
    match = PATTERN.search(path.stem)
    return (match.group(1), int(match.group(2))) if match else (path.stem, 0)


def split_class(files, rng):
    groups = defaultdict(list)
    for path in klipp:
        groups[recording_info(path)[0]].append(path)
    targets = np.array([0.70, 0.15, 0.15]) * len(files)
    names = ('train','val','test')
    out = {name: [] for name in names}
    if len(groups) < 3:
        ordered = sorted(files, key=recording_info)
        n_train = round(targets[0]); n_val = max(1, round(targets[1]))
        out['train'] = ordered[:n_train]
        out['val'] = ordered[n_train:n_train+n_val]
        out['test'] = ordered[n_train+n_val:]
        return out, False
    batches = list(groups.values())
    rng.shuffle(batches)
    batches.sort(key=len, reverse=True)
    for group in batches:
        dest = max(names, key=lambda n: targets[names.index(n)] - len(out[n]))
        out[dest].extend(group)
    return out, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = __import__('random').Random(args.seed)
    files_by_split = {name: [] for name in ('train','val','test')}
    caveats = []
    for class_dir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob('*.wav'))
        splits, independent = split_class(files, rng)
        for name, split_files in splits.items():
            files_by_split[name].extend((f, class_dir.name) for f in split_files)
        n_recordings = len({recording_info(f)[0] for f in files})
        print(class_dir.name, 'klipp:', len(files), 'inspelningar:', n_recordings,
              'uppdelning:', {k: len(v) for k,v in splits.items()})
        if not independent:
            caveats.append(f'{class_dir.name}: bara {n_recordings} ursprungsinspelning(ar); test och träning kan komma från samma inspelning')
    data = {}
    for name, pairs in files_by_split.items():
        X = np.stack([features_from_spectrogram(spectrogram_from_audio(path)) for path,_ in pairs])
        y = np.array([label for _,label in pairs])
        data[name] = (X, y)
        print(f'{name}: {len(y)} klipp')
    classifier = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=2,
                                      class_weight='balanced', random_state=args.seed, n_jobs=-1)
    classifier.fit(*data['train'])
    results = {'model': 'ExtraTrees med egenskaper från mel spektrogram',
               'seed': args.seed, 'data_source': str(args.data),
               'classes': list(map(str, classifier.classes_)),
               'counts': {k: dict(Counter(y)) for k,(_,y) in data.items()},
               'caveats': caveats,
               'evaluation': {}}
    for name in ('val', 'test'):
        X,y = data[name]
        pred = classifier.predict(X)
        report = classification_report(y, pred, labels=classifier.classes_,
                                       output_dict=True, zero_division=0)
        results['evaluation'][name] = {
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
            'confusion_matrix': confusion_matrix(y,pred,labels=classifier.classes_).tolist(),
            'classification_report': report,
        }
        print(f'{name} träffsäkerhet: {results["evaluation"][name]["accuracy"]:.3f}; balanced träffsäkerhet: {results["evaluation"][name]["balanced_accuracy"]:.3f}')
        print('Förväxlingsmatris, rader=facit och kolumner=förutsägelse:', classifier.classes_)
        print(np.asarray(results['evaluation'][name]['confusion_matrix']))
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.output/'model.joblib', compress=3)
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Sparade:', args.output/'model.joblib', args.output/'metrics.json')

if __name__ == '__main__':
    main()
