#!/usr/bin/env python3
"""Full DFlash2 fine-tuning on locally captured Q5 project traces."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from torch.nn import functional as F
from gguf import GGUFReader,GGUFWriter,GGUFValueType
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from gguf_utils import tensor_name,read_rows,LAYERS
from common import metric_guard
from amp_step import amp_train_step
from checkpoint_seed import validate_seed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from zmodel import DFlash2DraftModel

def load_samples(run):
 expected={r['id']:r for r in map(json.loads,(run/'data/all.jsonl').read_text().splitlines())}
 samples=[]
 for file in sorted((run/'features').glob('*.json')):
  sample=json.loads(file.read_text());reference=expected[sample['id']]
  assert sample['prompt']==reference['prompt'] and sample['source_sha256']==reference['source_sha256']
  shape=sample['feature_shape'];assert shape[0]==len(sample['input_ids']) and shape[1:]==[8,2048]
  suffix='.f16' if sample['feature_dtype']=='float16' else '.f32'
  dtype=np.float16 if suffix=='.f16' else np.float32
  feature=file.with_suffix(suffix)
  assert feature.stat().st_size==shape[0]*16384*np.dtype(dtype).itemsize
  if shape[0]-sample['prompt_n']<9:continue
  sample['feature_path']=feature
  sample['feature_numpy_dtype']=dtype
  samples.append(sample)
 return samples

def copy_feature_rows(sample,begin,end):
 # Keep descriptors bounded even when the dataset has thousands of feature files.
 mapped=np.memmap(sample['feature_path'],dtype=sample['feature_numpy_dtype'],mode='r',
                  shape=(sample['feature_shape'][0],16384))
 try:return np.array(mapped[begin:end],copy=True)
 finally:mapped._mmap.close()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--run-dir',type=Path,required=True)
 parser.add_argument('--steps',type=int,default=0)
 parser.add_argument('--state-dir',type=Path,required=True)
 parser.add_argument('--plan-hash',required=True)
 parser.add_argument('--seed-checkpoint',type=Path)
 parser.add_argument('--seed-plan-hash')
 parser.add_argument('--seed-step',type=int)
 parser.add_argument('--stage-key',required=True)
 parser.add_argument('--target-model',type=Path,required=True)
 parser.add_argument('--eval-only',action='store_true')
 parser.add_argument('--batch-size',type=int,default=4)
 parser.add_argument('--context',type=int,default=384)
 parser.add_argument('--learning-rate',type=float,default=1e-5)
 parser.add_argument('--initial-draft',type=Path,required=True)
 parser.add_argument('--output-name',default='draft-candidate-F16.gguf')
 parser.add_argument('--model-name',default='DFlash-Coach-candidate')
 parser.add_argument('--pool-field',default='project')
 parser.add_argument('--min-train',type=int,default=490)
 parser.add_argument('--min-eval',type=int,default=55)
 parser.add_argument('--min-total',type=int,default=550)
 parser.add_argument('--validate-data-only',action='store_true')
 args=parser.parse_args();run=args.run_dir;out=args.state_dir;out.mkdir(exist_ok=True)
 torch.set_num_threads(2);random.seed(731);torch.manual_seed(731)
 def emit(event,**values):
  row={'event':event,**values}
  print(json.dumps(row),flush=True)
  p=run/'evidence/training-progress.json'
  temp=p.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2));temp.replace(p)
 samples=load_samples(run)
 train=[s for s in samples if s['split']=='train'];valid=[s for s in samples if s['split']=='eval']
 assert len(samples)>=args.min_total and len(valid)>=args.min_eval and len(train)>=args.min_train,(len(train),len(valid))
 assert train and valid
 assert {(s['project'],s['source_path']) for s in train}.isdisjoint((s['project'],s['source_path']) for s in valid)
 if args.validate_data_only:
  before_fds=len(os.listdir('/proc/self/fd'));peak=before_fds
  for sample in samples:
   value=copy_feature_rows(sample,0,min(2,sample['feature_shape'][0]))
   assert np.isfinite(value).all(),sample['id']
   peak=max(peak,len(os.listdir('/proc/self/fd')))
  after_fds=len(os.listdir('/proc/self/fd'))
  assert after_fds<=before_fds+1,(before_fds,after_fds)
  emit('DATA_VALIDATED',samples=len(samples),train_examples=len(train),eval_examples=len(valid),
       before_fds=before_fds,after_fds=after_fds,peak_sampled_fds=peak)
  return
 device=torch.device('cuda:0')
 source=args.initial_draft
 reader=GGUFReader(str(source));tensors={t.name:t for t in reader.tensors}
 config=Qwen3Config(hidden_size=2048,intermediate_size=6144,num_hidden_layers=6,num_attention_heads=32,
  num_key_value_heads=8,head_dim=128,vocab_size=248320,rms_norm_eps=1e-6,
  max_position_embeddings=262144,sliding_window=4096,use_sliding_window=True,
  layer_types=['sliding_attention']*6,is_causal=False,num_target_layers=40,
  rope_parameters={'rope_type':'default','rope_theta':1e7},
  dflash_config={'block_size':16,'target_layer_ids':[x-1 for x in LAYERS],
   'mask_token_id':248077,'conv_kernel_size':2,'conv_group_size':16,'selector_rank':256,'selector_top_k':16})
 with torch.device('meta'):model=DFlash2DraftModel(config)
 state={}
 for name,param in model.state_dict().items():
  value=torch.from_numpy(np.array(tensors[tensor_name(name)].data,copy=True)).to(device=device,dtype=torch.float32)
  assert value.shape==param.shape,(name,value.shape,param.shape)
  state[name]=value
 model.load_state_dict(state,assign=True);del state
 model.rotary_emb=Qwen3RotaryEmbedding(config,device=device)
 model.requires_grad_(True)
 target=GGUFReader(str(args.target_model));tt={t.name:t for t in target.tensors}
 embedding=read_rows(tt['token_embd.weight'],device);head=read_rows(tt['output.weight'],device)
 def compute(batch,anchors,width):
  b=len(batch);ctx=args.context
  x=torch.zeros((b,ctx,16384),device=device,dtype=torch.float16)
  noise=torch.full((b,width),248077,device=device,dtype=torch.long)
  positions=torch.zeros((b,ctx+width),device=device,dtype=torch.long)
  attention=torch.ones((b,1,width,ctx+width),device=device,dtype=torch.bool)
  gold=[];previous=[]
  for index,(sample,anchor) in enumerate(zip(batch,anchors)):
   begin=max(0,anchor-ctx);length=anchor-begin;pad=ctx-length
   x[index,pad:]=torch.from_numpy(copy_feature_rows(sample,begin,anchor)).to(device=device,dtype=torch.float16)
   positions[index,pad:]=torch.arange(begin,anchor+width,device=device)
   attention[index,:,:,:pad]=False
   ids=sample['input_ids'];noise[index,0]=ids[anchor]
   gold.append(ids[anchor+1:anchor+width]);previous.append(ids[anchor:anchor+width-1])
  gold=torch.tensor(gold,device=device);previous=torch.tensor(previous,device=device)
  with torch.autocast('cuda',dtype=torch.float16):
   hidden=model(position_ids=positions,attention_mask=attention,noise_embedding=F.embedding(noise,embedding),target_hidden=x)
   logits=F.linear(hidden[:,1:],head).float()
   ce=F.cross_entropy(logits.reshape(-1,248320),gold.flatten())
   unary,candidates=logits.topk(16,dim=-1)
   selector=model.candidate_selector
   gate=selector.hidden_projection(hidden[:,1:]).float()
   scores=unary+((selector.predecessor_codebook(previous).float()*gate)[:,:,None,:]*selector.successor_codebook(candidates).float()).sum(-1)
   matches=candidates.eq(gold[:,:,None]);covered=matches.any(-1)
   selector_loss=F.cross_entropy(scores[covered],matches.long().argmax(-1)[covered]) if covered.any() else scores.sum()*0
   loss=ce+0.5*selector_loss
  chosen=candidates.gather(-1,scores.argmax(-1,keepdim=True)).squeeze(-1)
  correct=chosen.eq(gold)
  prefix=correct.long().cumprod(-1).sum(-1).float()
  return loss,{'ce':float(ce.detach()),'selector_ce':float(selector_loss.detach()),
   'accuracy':float(correct.float().mean()),'prefix_tokens':float(prefix.mean()),'top16_recall':float(covered.float().mean())}
 def evaluate(full=True):
  model.eval();chosen=valid if full else valid[:min(16,len(valid))]
  records=[]
  with torch.no_grad():
   for start in range(0,len(chosen),args.batch_size):
    batch=chosen[start:start+args.batch_size]
    for fraction in (0.25,0.7):
     anchors=[s['prompt_n']+int((len(s['input_ids'])-s['prompt_n']-8)*fraction) for s in batch]
     loss,metrics=compute(batch,anchors,8);records.append({'loss':float(loss.detach()),**metrics,'samples':len(batch)})
  weights=sum(r['samples'] for r in records)
  return {key:sum(r[key]*r['samples'] for r in records)/weights for key in records[0] if key!='samples'}
 # Validate after restoring the resumable model below.
 fc=[];body=[]
 for name,param in model.named_parameters():(fc if name.startswith(('fc.','hidden_norm.')) else body).append(param)
 optimizer=torch.optim.AdamW([{'params':body,'lr':args.learning_rate,'base_lr':args.learning_rate},
  {'params':fc,'lr':args.learning_rate*3,'base_lr':args.learning_rate*3}],weight_decay=0.01,foreach=True)
 scaler=torch.amp.GradScaler('cuda',init_scale=1024)
 started=time.monotonic();history=[];bad=0;amp_retries=0
 start_step=1
 checkpoint=out/'last-state.pt'
 seeding=not checkpoint.exists() and args.seed_checkpoint is not None
 if checkpoint.exists() or seeding:
  saved=torch.load(args.seed_checkpoint if seeding else checkpoint,map_location='cpu',weights_only=False)
  if seeding:validate_seed(saved,args.seed_plan_hash,args.seed_step)
  elif saved['plan_hash']!=args.plan_hash:raise RuntimeError('Checkpoint belongs to another frozen plan')
  model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
  random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
  torch.cuda.set_rng_state_all(saved['cuda_rng'])
  start_step=saved['step']+1
  best_step=saved['best_step'];best_metrics=saved['best_metrics'];best_loss=best_metrics['loss']
  best_state=saved['best_model']
  bad=saved.get('bad',0);baseline=saved['baseline']
  completed_stage=saved['stage_key']
  # A retry after a committed shard must not train it a second time.
  if completed_stage==args.stage_key and not args.eval_only:
   (out/'training.json').write_text(json.dumps(saved['result'],indent=2))
   emit('ALREADY_COMMITTED',stage_key=args.stage_key,step=saved['step']);return
  del saved
  emit('RESUMED',step=start_step)
 before=evaluate()
 if any(not math.isfinite(float(value)) for value in before.values()):raise RuntimeError('Non-finite initial validation')
 emit('VALIDATION_BEFORE',metrics=before,train_examples=len(train),eval_examples=len(valid))
 if args.eval_only:
  emit('EVALUATED',metrics=before);return
 if start_step==1:
  best_step=0;best_metrics=before;best_loss=before['loss'];baseline=before
 # Every accepted training sample is visited once in this shard, in a seeded order.
 ordered=sorted(train,key=lambda row:hashlib.sha256((args.stage_key+row['id']).encode()).hexdigest())
 batches=[ordered[i:i+args.batch_size] for i in range(0,len(ordered),args.batch_size)]
 shard_steps=len(batches)
 # One complete shard is trained per invocation; no global step/epoch ceiling.
 def save_best(step,metrics):
  nonlocal best_state
  best_state={k:v.detach().half().cpu().clone() for k,v in model.state_dict().items()}
 if start_step==1:save_best(0,before)
 for local_step,batch in enumerate(batches):
  step=start_step+local_step
  model.train()
  width=8 if step%4==0 else 4
  anchors=[random.randint(s['prompt_n'],len(s['input_ids'])-width-1) for s in batch]
  scale=min(1.,step/100)
  for index,group in enumerate(optimizer.param_groups):
   group['base_lr']=args.learning_rate*(3 if index==1 else 1)
   group['lr']=group['base_lr']*scale
  def overflow_event(attempt,old_scale,new_scale):
   emit('AMP_OVERFLOW_RETRY',step=step,attempt=attempt,scale_before=old_scale,
        scale_after=new_scale,batch_ids=[sample['id'] for sample in batch])
  loss,metrics,retries=amp_train_step(lambda:compute(batch,anchors,width),model.parameters(),
                                     optimizer,scaler,on_overflow=overflow_event)
  amp_retries+=retries
  row={'step':step,'loss':float(loss.detach()),**metrics,'elapsed_seconds':time.monotonic()-started}
  if step<=20 or step%50==0:
   history.append(row);elapsed=time.monotonic()-started
   emit('TRAIN_STEP',**row,epoch_step_estimate=args.steps,estimated_remaining_seconds=elapsed/(step-start_step+1)*max(0,args.steps-step),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
 # Shards commit atomically only after every sample and validation have completed.
 last=evaluate();emit('VALIDATION',step=step,metrics=last)
 metric_guard(baseline,last)
 if last['loss']<best_loss:
  best_loss=last['loss'];best_metrics=last;best_step=step;bad=0;save_best(step,last)
 else:bad+=1
 # Artifacts are written before the single checkpoint commit. A failed shard can be replayed.
 current_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(best_state)
 by_name={tensor_name(name):value for name,value in model.state_dict().items()}
 writer=GGUFWriter(str(out/args.output_name),'dflash')
 for key,field in reader.fields.items():
  if key.startswith('GGUF.') or key=='general.architecture':continue
  value=field.contents()
  if key=='general.name':value=args.model_name
  if field.types[0]==GGUFValueType.ARRAY:writer.add_array(key,value)
  else:writer.add_key_value(key,value,field.types[0])
 for tensor in reader.tensors:
  value=by_name[tensor.name].detach().cpu().numpy().astype(tensor.data.dtype)
  writer.add_tensor(tensor.name,value,raw_dtype=tensor.tensor_type)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 result={'stage_key':args.stage_key,'baseline':baseline,'steps':step,'best_step':best_step,'before':before,'last':last,'best':best_metrics,
  'seconds':time.monotonic()-started,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
  'trainable_parameters':sum(p.numel() for p in model.parameters()),'train_examples':len(train),'eval_examples':len(valid),
  'initial_draft':str(source),'learning_rate':args.learning_rate,
  'amp_overflow_retries':amp_retries,'amp_final_scale':scaler.get_scale(),'pool_field':args.pool_field,'accepted_training_ids':[s['id'] for s in ordered],
  'method':'full draft fine-tuning, FP32 master weights, FP16 autocast; frozen Q5 target embedding and LM head','history':history}
 (out/'training.json').write_text(json.dumps(result,indent=2))
 temp=out/'last-state.tmp'
 torch.save({'step':step,'stage_key':args.stage_key,'plan_hash':args.plan_hash,
  'model':current_state,'best_model':best_state,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
  'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
  'best_step':best_step,'best_metrics':best_metrics,'baseline':baseline,'bad':bad,'result':result},temp)
 temp.replace(checkpoint)
 emit('TRAIN_DONE',**{k:v for k,v in result.items() if k not in ('history','accepted_training_ids')})

if __name__=='__main__':main()
