import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from download_dataset import public_input,split_validation


class DatasetTests(unittest.TestCase):
    def row(self,i,split='validation'):
        return {'id':f'synthetic_{i}','project':'example','source_path':f'file_{i//2}.txt',
                'source_sha256':hashlib.sha256(f'source:{i//2}'.encode()).hexdigest(),
                'sampling_group':'example/en','prompt':f'Synthetic prompt {i}.','language':'en',
                'split':split,'completion':'This answer must not become a target label.',
                'provenance':{'source':'synthetic'}}

    def test_downloader_discards_supplied_answers(self):
        row=public_input(self.row(0),'train')
        self.assertNotIn('completion',row);self.assertNotIn('provenance',row)
        self.assertEqual(row['prompt_tokens'],0)

    def test_validation_source_families_stay_together(self):
        left,right=split_validation([self.row(i) for i in range(8)])
        self.assertEqual(len(left)+len(right),8)
        self.assertTrue({r['source_path'] for r in left}.isdisjoint(r['source_path'] for r in right))

    def test_prepare_builds_an_immutable_plan_without_gpu_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input';source.mkdir()
            config={'dataset':str(source),'run_dir':str(root/'run'),'target_model':str(root/'target.gguf'),
                'initial_draft':str(root/'draft.gguf'),'capture_binary':str(root/'capture'),
                'tokenize_binary':str(root/'tokenize'),'server_binary':str(root/'server'),
                'server_argv':['{server_binary}'],'validation_monitor_rows':16,'benchmark_test_rows':2,
                'max_shard_rows':4,'shard_feature_budget_gib':1,'capture_context':512,'max_new_tokens':32}
            for name in ('target.gguf','draft.gguf','capture','tokenize','server'):(root/name).write_bytes(b'SYNTHETIC TEST FILE')
            for offset,(split,count) in enumerate((('train',8),('validation',16),('test',2))):
                rows=[]
                for i in range(count):
                    row=public_input(self.row(offset*100+i,split),split)
                    row['prompt_tokens']=16;rows.append(row)
                (source/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            f=root/'config.json';f.write_text(json.dumps(config))
            args=[sys.executable,str(ROOT/'coach.py'),'prepare','--config',str(f)]
            subprocess.run(args,capture_output=True,text=True,check=True)
            plan=json.loads((root/'run/plan.json').read_text())
            self.assertEqual(plan['train_tasks'],8);self.assertEqual(len(plan['shards']),2)
            self.assertEqual(plan['tokenization_rows'],26)
            retry=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(retry.returncode,0)
            self.assertFalse((root/'run/evidence/run-state.json').exists())


if __name__=='__main__':unittest.main()
