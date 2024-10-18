from typing import List, NamedTuple
import torch
from safetensors import safe_open
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

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
    filtered_dict = {k.replace(f"{encoder_name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{encoder_name}.")}
    
    layer_weights = []
    for i in range(n_layers):
        layer_weights.append(LayerWeights(**{
            f"{w}{('_b' if 'bias' in k else '')}": filtered_dict[f"encoder.layers.{i}.{k}"]
            for k, w in [
                ("self_attn.q_proj.weight", "wq"), ("self_attn.k_proj.weight", "wk"),
                ("self_attn.v_proj.weight", "wv"), ("self_attn.out_proj.weight", "wo"),
                ("mlp.0.weight", "w1"), ("mlp.2.weight", "w2"),
                ("layer_norm1.weight", "attention_norm"), ("layer_norm2.weight", "ffn_norm"),
                ("self_attn.q_proj.bias", "wq"), ("self_attn.k_proj.bias", "wk"),
                ("self_attn.v_proj.bias", "wv"), ("self_attn.out_proj.bias", "wo"),
                ("mlp.0.bias", "w1"), ("mlp.2.bias", "w2"),
                ("layer_norm1.bias", "attention_norm"), ("layer_norm2.bias", "ffn_norm")
            ]
        }))

    if encoder_name == "text_model":
        return TextTransformerWeights(
            tok_embeddings=filtered_dict['token_embedding.weight'],
            pos_embeddings=filtered_dict["position_embedding.weight"],
            post_norm=filtered_dict['final_layer_norm.weight'],
            post_norm_b=filtered_dict['final_layer_norm.bias'],
            layer_weights=layer_weights
        )
    elif encoder_name == "vision_model":
        return VisionTransformerWeights(
            pos_embeddings=filtered_dict['position_embedding.weight'],
            cls_embedding=filtered_dict['class_embedding'], 
            patch_embeddings=filtered_dict['patch_embedding.weight'],
            pre_norm=filtered_dict['pre_layernorm.weight'],
            pre_norm_b=filtered_dict['pre_layernorm.bias'], 
            post_norm=filtered_dict['post_layernorm.weight'],
            post_norm_b=filtered_dict['post_layernorm.bias'], 
            layer_weights=layer_weights
        )

def load_owlv2_weights(state_dict):
    text_weights = load_encoder_weights(state_dict, encoder_name="text_model")
    vision_weights = load_encoder_weights(state_dict, encoder_name="vision_model")

    head_configs = {
        "box_head": ["dense0", "dense1", "dense2"],
        "class_head": ["dense0"],
        "objectness_head": ["dense0", "dense1", "dense2"]
    }

    heads = {}
    for i, (head_name, layers) in enumerate(head_configs.items()):
        head_dict = {f"w{i+1}": state_dict[f"{head_name}.{layer}.weight"] for i, layer in enumerate(layers)}
        head_dict.update({f"w{i+1}_b": state_dict[f"{head_name}.{layer}.bias"] for i, layer in enumerate(layers)})
        
        if head_name == "class_head":
            head_dict.update({
                "logit_scale": state_dict[f"{head_name}.logit_scale.weight"],
                "logit_scale_b": state_dict[f"{head_name}.logit_scale.bias"],
                "logit_shift": state_dict[f"{head_name}.logit_shift.weight"],
                "logit_shift_b": state_dict[f"{head_name}.logit_shift.bias"]
            })
            head_dict.update({k: None for k in ["w2","w2_b", "w3","w3_b"]})
        else:
            head_dict.update({k: None for k in ["logit_scale", "logit_scale_b", "logit_shift", "logit_shift_b"]})
        
        heads[head_name] = HeadLayerWeights(**head_dict)

    return OWLv2Weights(
        logit_scale=state_dict['logit_scale'],
        text_proj=state_dict['text_projection.weight'],
        vision_proj=state_dict['visual_projection.weight'],
        layer_norm=state_dict['layer_norm.weight'],
        layer_norm_b=state_dict['layer_norm.bias'],
        vision_weights=vision_weights,
        text_weights=text_weights,
        box_head=heads['box_head'],
        class_head=heads['class_head'],
        objectness_head=heads['objectness_head']
    )

if __name__ == "__main__":
    ckpt_dir: Path = Path('weights/model.safetensors')
    with safe_open(ckpt_dir, framework="pt") as f:
        state_dict = {k.replace("owlv2.", "").replace(".embeddings", ""): f.get_tensor(k) for k in f.keys()}
    state_dict = {k.replace("mlp.fc1", "mlp.0").replace("mlp.fc2", "mlp.2"): v for k, v in state_dict.items()}

    print([e for e in list(state_dict.keys()) if "model" not in e])
    load_owlv2_weights(state_dict)