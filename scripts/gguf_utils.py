"""GGUF tensor mapping and chunked dequantization for the supported backend."""
import numpy as np
import torch
from gguf import dequantize

LAYERS=[1,6,11,16,22,27,32,37]

def tensor_name(name):
 direct={'fc.weight':'fc.weight','hidden_norm.weight':'enc.output_norm.weight','norm.weight':'output_norm.weight',
 'candidate_selector.predecessor_codebook.weight':'selector_predecessor.weight',
 'candidate_selector.successor_codebook.weight':'selector_successor.weight',
 'candidate_selector.hidden_projection.weight':'selector_hidden.weight'}
 if name in direct:return direct[name]
 parts=name.split('.'); assert parts[0]=='layers',name
 tail='.'.join(parts[2:])
 table={'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'ffn_norm.weight',
 'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight',
 'self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
 'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
 'mlp.gate_proj.weight':'ffn_gate.weight','mlp.up_proj.weight':'ffn_up.weight','mlp.down_proj.weight':'ffn_down.weight',
 'attention_conv.base_kernel':'attn_conv_base','attention_conv.kernel_projection.weight':'attn_conv_proj.weight',
 'mlp_conv.base_kernel':'ffn_conv_base','mlp_conv.kernel_projection.weight':'ffn_conv_proj.weight'}
 return f'blk.{parts[1]}.'+table[tail]


def read_rows(tensor,device):
 rows=int(tensor.shape[1]); cols=int(tensor.shape[0])
 out=torch.empty((rows,cols),dtype=torch.float16,device=device)
 for begin in range(0,rows,1024):
  part=dequantize(tensor.data[begin:begin+1024],tensor.tensor_type)
  out[begin:begin+len(part)].copy_(torch.from_numpy(np.array(part,copy=True)).to(device=device,dtype=torch.float16))
 return out

