#!/usr/bin/env python3
"""Download a pinned public dataset example. It never uploads files."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request
from common import atomic_json,write_rows

ROOT=Path(__file__).resolve().parents[1]


def get_rows(profile,name):
    url=f'https://huggingface.co/datasets/{profile["repository"]}/resolve/{profile["revision"]}/{name}'
    with urllib.request.urlopen(url,timeout=120) as response:
        content=response.read()
    rows=[json.loads(line) for line in content.decode().splitlines() if line.strip()]
    return rows,hashlib.sha256(content).hexdigest()


def public_input(row,split):
    # Completed answers and source/provenance payloads are not used as target labels.
    names=('id','project','source_path','source_sha256','sampling_group','prompt','language')
    out={name:row[name] for name in names}
    out.update(split=split,dataset_split=split,prompt_tokens=row.get('prompt_tokens',0),
        source_group=row.get('source_group',row['project']+':'+row['source_path']),
        length_bucket=row.get('length_bucket','unknown'))
    return out


def split_validation(rows):
    """Reserve complete source families when an example dataset has no test split."""
    families=defaultdict(list)
    for row in rows:families[(row['project'],row['source_path'])].append(row)
    keys=sorted(families,key=lambda k:hashlib.sha256(repr(k).encode()).hexdigest())
    if len(keys)<2:raise ValueError('Need at least two validation source families to reserve a test split')
    groups=[[],[]]
    for i,key in enumerate(keys):groups[i%2].extend(families[key])
    return groups


def main():
    profiles=json.loads((ROOT/'examples/huggingface-datasets.json').read_text())
    parser=argparse.ArgumentParser();parser.add_argument('--example',choices=profiles,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise SystemExit('Choose an empty output directory')
    profile=profiles[a.example];groups={};hashes={}
    for split,name in profile['files'].items():
        rows,sha=get_rows(profile,name);groups[split]=rows;hashes[name]=sha
    if 'test' not in groups:
        groups['validation'],groups['test']=split_validation(groups['validation'])
    counts={}
    for split,rows in groups.items():
        clean=[public_input(row,split) for row in rows];counts[split]=len(clean)
        write_rows(a.output/(split+'.jsonl'),clean)
    atomic_json(a.output/'SOURCES.json',{'repository':profile['repository'],'revision':profile['revision'],
        'downloaded_sha256':hashes,'example':a.example,'counts':counts,
        'note':'Only normalized prompt contexts are retained. Run tokenize-inputs before preparation.'})
    print(json.dumps({'status':'downloaded','counts':counts,'output':str(a.output)},indent=2))


if __name__=='__main__':main()
