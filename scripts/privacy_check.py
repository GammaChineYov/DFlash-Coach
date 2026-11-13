#!/usr/bin/env python3
"""Check the source tree for common publication mistakes without printing matches."""
import argparse
import json
from pathlib import Path
import re

PATTERNS={
    'private_key':re.compile(r'-----BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'access_token':re.compile(r'\b(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,})\b'),
    'personal_home':re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+'),
    'mounted_volume':re.compile(r'/Volumes/[A-Za-z0-9_.-]+'),
    'private_network':re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
    'device_uuid':re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{20,}'),
}
FORBIDDEN_SUFFIXES={'.gguf','.safetensors','.pt','.pth','.f16','.f32','.pem','.key','.sqlite','.db','.zip','.tar','.gz'}


def inspect(root):
    root=Path(root);findings=[];files=0
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(part in ('.git','.venv','__pycache__') for part in rel.parts) or not path.is_file():continue
        files+=1
        if path.is_symlink():findings.append({'file':str(rel),'kind':'symlink'});continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name=='.env' or path.name.endswith('.local.json'):
            findings.append({'file':str(rel),'kind':'private_or_binary_artifact'});continue
        if any(part in ('runs','models','data','verification') for part in rel.parts):
            findings.append({'file':str(rel),'kind':'runtime_artifact'});continue
        try:text=path.read_text()
        except UnicodeDecodeError:
            findings.append({'file':str(rel),'kind':'binary_file'});continue
        for name,pattern in PATTERNS.items():
            if pattern.search(text):findings.append({'file':str(rel),'kind':name})
    return {'status':'passed' if not findings else 'failed','files_checked':files,'findings':findings,
            'scope':'Static source check. It does not prove the absence of every identifying detail.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    report=inspect(a.root);print(json.dumps(report,indent=2))
    if report['findings']:raise SystemExit(1)


if __name__=='__main__':main()
