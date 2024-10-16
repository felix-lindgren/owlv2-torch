from typing import List, NamedTuple


import torch
import numpy as np


from safetensors import safe_open

from pathlib import Path

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

class LayerWeights(NamedTuple):
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor
  wq_b: torch.Tensor
  wk_b: torch.Tensor
  wv_b: torch.Tensor
  wo_b: torch.Tensor
  w1_b: torch.Tensor
  w2_b: torch.Tensor
  ffn_norm_b: torch.Tensor
  attention_norm_b: torch.Tensor

class TextTransformerWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  pos_embeddings: torch.Tensor
  post_norm: torch.Tensor
  post_norm_b: torch.Tensor
  layer_weights: List[LayerWeights]

class VisionTransformerWeights(NamedTuple):
  patch_embeddings: torch.Tensor
  pos_embeddings: torch.Tensor
  cls_embedding: torch.Tensor 
  pre_norm: torch.Tensor
  pre_norm_b: torch.Tensor
  post_norm: torch.Tensor
  post_norm_b: torch.Tensor
  layer_weights: List[LayerWeights]

class HeadLayerWeights(NamedTuple):
   w1: torch.Tensor
   w2: torch.Tensor
   w3: torch.Tensor
   w1_b: torch.Tensor
   w2_b: torch.Tensor
   w3_b: torch.Tensor

   logit_scale: torch.Tensor
   logit_scale_b: torch.Tensor 
   logit_shift: torch.Tensor
   logit_shift_b: torch.Tensor


class OWLv2Weights(NamedTuple):
  logit_scale: torch.Tensor
  text_proj: torch.Tensor 
  vision_proj: torch.Tensor
  layer_norm: torch.Tensor
  layer_norm_b: torch.Tensor

  class_head: HeadLayerWeights
  box_head: HeadLayerWeights 
  objectness_head: HeadLayerWeights

  vision_weights: VisionTransformerWeights
  text_weights: TextTransformerWeights


  

def load_encoder_weights(state_dict, encoder_name="vision_model", n_layers: int = 12):
    
    layer_weights = []
    state_dict = {k.replace(encoder_name+".",""):v for k,v in state_dict.items() if encoder_name in k}
    
    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            #"encoder.layers.0.self_attn.q_proj.bias"
            wq=state_dict[f'encoder.layers.{i}.self_attn.q_proj.weight'],
            wk=state_dict[f'encoder.layers.{i}.self_attn.k_proj.weight'],
            wv=state_dict[f'encoder.layers.{i}.self_attn.v_proj.weight'],
            wo=state_dict[f'encoder.layers.{i}.self_attn.out_proj.weight'],
            w1=state_dict[f'encoder.layers.{i}.mlp.0.weight'],
            w2=state_dict[f'encoder.layers.{i}.mlp.2.weight'],
            attention_norm=state_dict[f'encoder.layers.{i}.layer_norm1.weight'],
            ffn_norm=state_dict[f'encoder.layers.{i}.layer_norm2.weight'],
            wq_b=state_dict[f'encoder.layers.{i}.self_attn.q_proj.bias'],
            wk_b=state_dict[f'encoder.layers.{i}.self_attn.k_proj.bias'],
            wv_b=state_dict[f'encoder.layers.{i}.self_attn.v_proj.bias'],
            wo_b=state_dict[f'encoder.layers.{i}.self_attn.out_proj.bias'],
            w1_b=state_dict[f'encoder.layers.{i}.mlp.0.bias'],
            w2_b=state_dict[f'encoder.layers.{i}.mlp.2.bias'],
            attention_norm_b=state_dict[f'encoder.layers.{i}.layer_norm1.bias'],
            ffn_norm_b=state_dict[f'encoder.layers.{i}.layer_norm2.bias'],
        ))

    if encoder_name == "text_model":
        xfmr_weights = TextTransformerWeights(
        tok_embeddings=state_dict['token_embedding.weight'],
        pos_embeddings=state_dict["position_embedding.weight"],
        post_norm=state_dict['final_layer_norm.weight'],
        post_norm_b=state_dict['final_layer_norm.bias'],
        layer_weights=layer_weights
        )
    elif encoder_name == "vision_model":
       xfmr_weights = VisionTransformerWeights(
        pos_embeddings=state_dict['position_embedding.weight'],
        cls_embedding=state_dict['class_embedding'], 
        patch_embeddings=state_dict['patch_embedding.weight'],
        pre_norm=state_dict['pre_layernorm.weight'],
        pre_norm_b=state_dict['pre_layernorm.bias'], 
        post_norm=state_dict['post_layernorm.weight'],
        post_norm_b=state_dict['post_layernorm.bias'], 
        layer_weights=layer_weights
        ) 

    return xfmr_weights

def load_owlv2_weights(state_dict):
    text_weights = load_encoder_weights(state_dict=state_dict, encoder_name="text_model")
    vision_weights = load_encoder_weights(state_dict=state_dict, encoder_name="vision_model")

    box_head = HeadLayerWeights(
       w1=state_dict['box_head.dense0.weight'],
       w1_b=state_dict['box_head.dense0.bias'],
       w2=state_dict['box_head.dense1.weight'],
       w2_b=state_dict['box_head.dense1.bias'],
       w3=state_dict['box_head.dense2.weight'],
       w3_b=state_dict['box_head.dense2.bias'],
       logit_scale=None, 
       logit_scale_b=None, 
       logit_shift=None, 
       logit_shift_b=None, 
    )
    cls_head = HeadLayerWeights(
       w1=state_dict['class_head.dense0.weight'],
       w1_b=state_dict['class_head.dense0.bias'],
       w2=None,
       w2_b=None,
       w3=None,
       w3_b=None,
       logit_scale=state_dict['class_head.logit_scale.weight'], 
       logit_scale_b=state_dict['class_head.logit_scale.bias'], 
       logit_shift=state_dict['class_head.logit_shift.weight'], 
       logit_shift_b=state_dict['class_head.logit_shift.bias'], 
    )
    objectness_head = HeadLayerWeights(
       w1=state_dict['objectness_head.dense0.weight'],
       w1_b=state_dict['objectness_head.dense0.bias'],
       w2=state_dict['objectness_head.dense1.weight'],
       w2_b=state_dict['objectness_head.dense1.bias'],
       w3=state_dict['objectness_head.dense2.weight'],
       w3_b=state_dict['objectness_head.dense2.bias'],
       logit_scale=None, 
       logit_scale_b=None, 
       logit_shift=None, 
       logit_shift_b=None, 
    )

    return OWLv2Weights(
        logit_scale=state_dict['logit_scale'],
        text_proj=state_dict['text_projection.weight'],
        vision_proj=state_dict['visual_projection.weight'],
        layer_norm=state_dict['layer_norm.weight'],
        layer_norm_b=state_dict['layer_norm.bias'],
        vision_weights=vision_weights,
        text_weights=text_weights,
        box_head=box_head,
        class_head=cls_head,
        objectness_head=objectness_head
    )

   

if __name__ == "__main__":
    ckpt_dir: Path = Path('model.safetensors')
    state_dict = {}
    with safe_open("model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens

    print([e for e in list(state_dict.keys()) if "model" not in e])
    
    load_owlv2_weights(state_dict)



