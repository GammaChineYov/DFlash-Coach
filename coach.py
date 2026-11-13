#!/usr/bin/env python3
"""Portable entry point; demo/status/tests require only the standard library."""
import argparse
from pathlib import Path
import runpy
import sys

COMMANDS={'launch':'launch','demo':'demo','prepare':'prepare','verify-tokens':'verify_tokens','preflight':'preflight',
          'download-example':'download_dataset','tokenize-inputs':'tokenize_inputs','run':'run','benchmark':'benchmark','status':'status','privacy-check':'privacy_check'}


def main():
    parser=argparse.ArgumentParser(description='DFlash Coach: draft fine-tuning orchestration')
    parser.add_argument('command',choices=COMMANDS)
    args,remaining=parser.parse_known_args()
    scripts=Path(__file__).resolve().parent/'scripts'
    sys.path.insert(0,str(scripts));sys.argv=[str(scripts/(COMMANDS[args.command]+'.py')),*remaining]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
