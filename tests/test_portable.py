import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import load_config,command_argv
from mentor import read_key
from prepare import normalize
from privacy_check import inspect
import run as supervisor


class ConfigTests(unittest.TestCase):
    def test_paths_are_relative_to_config_not_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'configs/example.json';f.parent.mkdir()
            f.write_text(json.dumps({'run_dir':'../runs/test','dataset':'../inputs','server_argv':['binary','{target_model}'],
                                   'target_model':'../models/a.gguf'}))
            cfg=load_config(f)
            self.assertEqual(Path(cfg['run_dir']),Path(tmp).resolve()/'runs/test')
            self.assertEqual(command_argv(cfg)[1],str(Path(tmp).resolve()/'models/a.gguf'))

    def test_output_path_cannot_escape_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'config.json';f.write_text(json.dumps({'output_name':'../escape.gguf'}))
            with self.assertRaises(ValueError):load_config(f)

    def test_key_comes_from_environment_without_a_file(self):
        with patch.dict(os.environ,{'COACH_TEST_CREDENTIAL':'synthetic-placeholder'}):
            self.assertEqual(read_key({'key_env':'COACH_TEST_CREDENTIAL'}),'synthetic-placeholder')

    def test_missing_key_is_explicit(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):read_key({'key_env':'COACH_TEST_CREDENTIAL'})


class InputTests(unittest.TestCase):
    def row(self):
        return {'id':'example_1','project':'example','source_path':'file.txt','source_sha256':'a'*64,
                'prompt':'Synthetic example.','prompt_tokens':10,'language':'en','sampling_group':'demo/en'}

    def test_normalization_separates_validation_labels(self):
        row=normalize(self.row(),'validation')
        self.assertEqual(row['dataset_split'],'validation')
        self.assertEqual(row['source_group'],'example:file.txt')

    def test_path_traversal_in_sample_id_is_rejected(self):
        row=self.row();row['id']='../outside'
        with self.assertRaises(ValueError):normalize(row,'train')

    def test_invalid_hash_is_rejected(self):
        row=self.row();row['source_sha256']='not-a-hash'
        with self.assertRaises(ValueError):normalize(row,'train')


class ServiceTests(unittest.TestCase):
    def test_no_service_is_touched_by_default(self):
        with patch('run.subprocess.run') as command:
            supervisor.service_control('stop',None)
            command.assert_not_called()

    def test_user_service_needs_no_sudo(self):
        with patch('run.subprocess.run') as command:
            supervisor.service_control('stop','example-inference.service')
            command.assert_called_once_with(['/usr/bin/systemctl','--user','stop','example-inference.service'],check=True)

    def test_restore_marker_clears_only_after_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'marker';marker.write_text('restore')
            with patch('run.stop_child'),patch('run.service_control'),patch('run.local_api',return_value={'status':'ok'}):
                supervisor.restore({'production_service':'example-inference.service'},marker)
            self.assertFalse(marker.exists())

    def test_restore_failure_retains_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'marker';marker.write_text('restore')
            with patch('run.stop_child'),patch('run.service_control',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):supervisor.restore({'production_service':'example-inference.service'},marker)
            self.assertTrue(marker.exists())

    def test_training_requires_launch_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'config.json';f.write_text(json.dumps({'run_dir':tmp}))
            with patch.object(sys,'argv',['run.py','--config',str(f)]),patch('run.subprocess.run') as command:
                with self.assertRaisesRegex(SystemExit,'Prepared only'):supervisor.main()
                command.assert_not_called()


class DemoTests(unittest.TestCase):
    def test_demo_runs_offline_and_does_not_claim_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'demo'
            result=subprocess.run([sys.executable,str(ROOT/'coach.py'),'demo','--run-dir',str(directory)],capture_output=True,text=True,check=True)
            report=json.loads(result.stdout)
            self.assertFalse(report['training_performed']);self.assertEqual(report['network_requests'],0)
            state=json.loads((directory/'evidence/run-state.json').read_text())
            self.assertEqual(len(state['committed']),3)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run([sys.executable,str(ROOT/'coach.py'),'demo','--run-dir',str(directory)],capture_output=True,check=True)


class PrivacyTests(unittest.TestCase):
    def test_empty_source_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:self.assertFalse(inspect(tmp)['findings'])

    def test_credential_file_and_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'.env').write_text('EMPTY=')
            (Path(tmp)/'draft.gguf').write_bytes(b'not-a-model')
            self.assertEqual(len(inspect(tmp)['findings']),2)

    def test_private_home_is_reported_without_its_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            value='/'+'home'+'/'+'example_private_user'
            (Path(tmp)/'file.txt').write_text(value)
            report=inspect(tmp)
            self.assertEqual(report['findings'][0]['kind'],'personal_home')
            self.assertNotIn(value,json.dumps(report))


if __name__=='__main__':unittest.main()
