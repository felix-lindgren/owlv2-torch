import torch
import torch.nn.functional as F
import numpy as np
import utils
from PIL import Image
from hf_version.modeling_owlv2 import Owlv2ForObjectDetection
from hf_version.processing_owlv2 import Owlv2Processor
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from torch_version.owlv2 import OwlV2
from torch_func.owlv2 import encode_text, encode_vision, attention, feed_forward, create_mask, build_attn_mask, combine_masks, pad_sequences
from torch_func.owlv2_config import OWLV2_B16, ModelParams
from torch_func.owlv2_weights import load_owlv2_weights, OWLv2Weights
from safetensors import safe_open


def test_attention(torch_model: OwlV2, weights: OWLv2Weights, model_params: ModelParams, device="cuda", layer_idx=0):
    x = torch.randn(1, 196, 768).to(device)
    with torch.no_grad():
        #pt_y, _ = torch_model.vision_model.encoder.layers[layer_idx].self_attn.forward(x, None)
        bsz, tgt_len, _ = x.size()
        # project the hidden states to the query, key, and value states
        query_states = torch_model.vision_model.encoder.layers[layer_idx].self_attn.q_proj(x) #* torch_model.vision_model.encoder.layers[layer_idx].self_attn.scale
        key_states = torch_model.vision_model.encoder.layers[layer_idx].self_attn._shape(torch_model.vision_model.encoder.layers[layer_idx].self_attn.k_proj(x), -1, bsz)
        value_states = torch_model.vision_model.encoder.layers[layer_idx].self_attn._shape(torch_model.vision_model.encoder.layers[layer_idx].self_attn.v_proj(x), -1, bsz)
        query_states = torch_model.vision_model.encoder.layers[layer_idx].self_attn._shape(query_states, tgt_len, bsz)

        # compute the attention weights
        sdpa_attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=torch_model.vision_model.encoder.layers[layer_idx].self_attn.dropout if torch_model.vision_model.encoder.layers[layer_idx].self_attn.training else 0.0,
            scale=torch_model.vision_model.encoder.layers[layer_idx].self_attn.scale
        )
        sdpa_attn_output_ = sdpa_attn_output
        sdpa_attn_output = sdpa_attn_output.transpose(1, 2)
        sdpa_attn_output = sdpa_attn_output.reshape(bsz, tgt_len, torch_model.vision_model.encoder.layers[layer_idx].self_attn.embed_dim)
        attn_output = torch_model.vision_model.encoder.layers[layer_idx].self_attn.out_proj(sdpa_attn_output)
        pt_y = attn_output
    with torch.no_grad():
        layer_weights = weights.vision_weights.layer_weights[layer_idx]
        xq = F.linear(x, layer_weights.wq, bias=layer_weights.wq_b).view(bsz, -1 ,model_params.vision_encoder.n_heads, model_params.vision_encoder.head_dim).transpose(1,2).contiguous()
        xk = F.linear(x, layer_weights.wk, bias=layer_weights.wk_b).view(bsz, -1 ,model_params.vision_encoder.n_heads, model_params.vision_encoder.head_dim).transpose(1,2).contiguous()
        xv = F.linear(x, layer_weights.wv, bias=layer_weights.wv_b).view(bsz, -1 ,model_params.vision_encoder.n_heads, model_params.vision_encoder.head_dim).transpose(1,2).contiguous()

        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=None,
            dropout_p=0.0,
            scale=model_params.vision_encoder.head_dim**-0.5
        )

        output = output.transpose(1,2)
        output = output.reshape(bsz, -1, model_params.vision_encoder.dim)
        output = F.linear(output, layer_weights.wo, layer_weights.wo_b)
        func_y = output

    print(f"Vision attention{layer_idx} match:", torch.allclose(func_y, pt_y, atol=1e-4))
    #print(pt_y, func_y)
    
def test_attention_masked(torch_model: OwlV2, weights: OWLv2Weights, model_params: ModelParams, device="cuda", layer_idx=0):
    
    seqlen = 8
    x = torch.rand(1,seqlen,512).to(device)
    seq_ = torch.arange(seqlen).unsqueeze(0).tolist()
    seq = pad_sequences(seq_)
    pad_mask = create_mask(seq_)
    causal_mask = build_attn_mask(1, seqlen, 0)
    attn_mask = combine_masks(pad_mask, causal_mask)

    with torch.no_grad():
        #pt_y, _ = torch_model.vision_model.encoder.layers[layer_idx].self_attn.forward(x, None)
        bsz, tgt_len, _ = x.size()
        # project the hidden states to the query, key, and value states
        query_states = torch_model.text_model.encoder.layers[layer_idx].self_attn.q_proj(x) #* torch_model.vision_model.encoder.layers[layer_idx].self_attn.scale
        key_states = torch_model.text_model.encoder.layers[layer_idx].self_attn._shape(torch_model.text_model.encoder.layers[layer_idx].self_attn.k_proj(x), -1, bsz)
        value_states = torch_model.text_model.encoder.layers[layer_idx].self_attn._shape(torch_model.text_model.encoder.layers[layer_idx].self_attn.v_proj(x), -1, bsz)
        query_states = torch_model.text_model.encoder.layers[layer_idx].self_attn._shape(query_states, tgt_len, bsz)

        # compute the attention weights
        sdpa_attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=torch_model.text_model.encoder.layers[layer_idx].self_attn.dropout if torch_model.text_model.encoder.layers[layer_idx].self_attn.training else 0.0,
            scale=torch_model.text_model.encoder.layers[layer_idx].self_attn.scale
        )
        sdpa_attn_output_ = sdpa_attn_output
        sdpa_attn_output = sdpa_attn_output.transpose(1, 2)
        sdpa_attn_output = sdpa_attn_output.reshape(bsz, tgt_len, torch_model.text_model.encoder.layers[layer_idx].self_attn.embed_dim)
        attn_output = torch_model.text_model.encoder.layers[layer_idx].self_attn.out_proj(sdpa_attn_output)
        pt_y = attn_output
    with torch.no_grad():
        layer_weights = weights.text_weights.layer_weights[layer_idx]
        xq = F.linear(x, layer_weights.wq, bias=layer_weights.wq_b).view(bsz, -1 ,model_params.text_encoder.n_heads, model_params.text_encoder.head_dim).transpose(1,2).contiguous()
        xk = F.linear(x, layer_weights.wk, bias=layer_weights.wk_b).view(bsz, -1 ,model_params.text_encoder.n_heads, model_params.text_encoder.head_dim).transpose(1,2).contiguous()
        xv = F.linear(x, layer_weights.wv, bias=layer_weights.wv_b).view(bsz, -1 ,model_params.text_encoder.n_heads, model_params.text_encoder.head_dim).transpose(1,2).contiguous()

        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=model_params.text_encoder.head_dim**-0.5
        )

        output = output.transpose(1,2)
        output = output.reshape(bsz, -1, model_params.text_encoder.dim)
        output = F.linear(output, layer_weights.wo, layer_weights.wo_b)
        func_y = output

    print(f"Text attention{layer_idx} match:", torch.allclose(func_y, pt_y, atol=1e-4))
    #print(pt_y, func_y)
def test_vision_encoder_layer(torch_model, weights: OWLv2Weights, model_params: ModelParams, device="cuda", layer_idx=0):
    x_ = torch.randn(1, 196, 768).to(device)
    with torch.no_grad():
        #pt_y = torch_model.vision_model.encoder.layers[layer_idx].forward(x)
        residual = x_
        
        x = torch_model.vision_model.encoder.layers[layer_idx].layer_norm1(x_)
        norm1_pt = x
        x, _ = torch_model.vision_model.encoder.layers[layer_idx].self_attn(x, None)
        attn_res_pt = x
        x = x + residual
        res1_pt = x
        residual = x
        x = torch_model.vision_model.encoder.layers[layer_idx].layer_norm2(x)
        norm2_pt = x
        x = torch_model.vision_model.encoder.layers[layer_idx].mlp(x)
        x = x + residual
        mlp_pt = x
        pt_y = x
    weights = weights.vision_weights
    with torch.no_grad():
        h = x_.clone()
        residual = h
        h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.layer_weights[layer_idx].attention_norm, bias=weights.layer_weights[layer_idx].attention_norm_b)
        h_attn  = attention(h, weights.layer_weights[layer_idx], model_params, attn_mask=None)
        h = residual + h_attn
        residual = h
        h = F.layer_norm(h, (model_params.vision_encoder.dim,), weight=weights.layer_weights[layer_idx].ffn_norm, bias=weights.layer_weights[layer_idx].ffn_norm_b)
        h = residual + feed_forward(h, weights.layer_weights[layer_idx])
        func_y = h

    print(f"Vision encoder layer{layer_idx} match:", torch.allclose(func_y, pt_y, atol=1e-4))
    #print(pt_y, func_y)

def test_vision_encoder_full(torch_model, weights: OWLv2Weights, model_params: ModelParams, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pt_y, unpooled = torch_model.vision_model.forward(inputs['pixel_values'])
    weights = weights.vision_weights
    with torch.no_grad():
        func_y, unpooled_func = encode_vision(inputs['pixel_values'], weights, model_params)
    print(pt_y.shape,unpooled.shape, func_y.shape)
    print(f"Vision encoder pooled match:", torch.allclose(func_y,pt_y, atol=1e-4))
    print(f"Vision encoder unpooled match:", torch.allclose(unpooled_func,unpooled, atol=1e-4))
    #print(pt_y, func_y)

def test_text_encoder_full(torch_model, weights: OWLv2Weights, model_params: ModelParams, device="cuda"):
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
    input_string = "a cat."
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=input_string, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pt_y, unpooled = torch_model.text_model.forward(inputs['input_ids'], inputs['attention_mask'])
    weights = weights.text_weights
    with torch.no_grad():
        input_shape = inputs['input_ids'].size()
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, torch.float32, "cpu")
        attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], torch.float32)
        attention_mask = attention_mask + causal_attention_mask
        func_y, unpooled_func = encode_text(inputs['input_ids'], weights, model_params, attn_mask=attention_mask)
    print(pt_y.shape,unpooled.shape, func_y.shape)
    print(f"Text encoder pooled match:", torch.allclose(func_y,pt_y, atol=1e-4))
    print(f"Text encoder unpooled match:", torch.allclose(unpooled_func,unpooled, atol=1e-4))
    
def test_vision_encoder(hf_model, torch_model, device="cuda"):
    x = torch.randn(1, 196, 768).to(device)
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model.encoder.forward(x, None, None, False, False, False)[0]

    with torch.no_grad():
        pt_y = torch_model.vision_model.encoder.forward(x)

    print(f"Vision encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_encoder_layer(hf_model, torch_model, device="cuda", layer_idx=0):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_embeddings = torch.rand(1, inputs['attention_mask'].shape[1], 512).to(device)
    with torch.no_grad():
        attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], input_embeddings.dtype)
        hf_y = hf_model.owlv2.text_model.encoder.layers[layer_idx].forward(input_embeddings, None, attention_mask)[0]
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model.encoder.layers[layer_idx].forward(input_embeddings, attention_mask)
    
    print(f"Text encoder layer{layer_idx} match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_embeddings(hf_model, torch_model, device="cuda"):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hf_y = hf_model.owlv2.text_model.embeddings(inputs['input_ids'])
    with torch.no_grad():
        seq_length = inputs['input_ids'].shape[-1] if inputs['input_ids'] is not None else inputs_embeds.shape[-2]
        position_ids = torch_model.text_model.position_ids[:, :seq_length]
        inputs_embeds = torch_model.text_model.token_embedding(inputs['input_ids'])
        position_embeddings = torch_model.text_model.position_embedding(position_ids)
        pt_y = inputs_embeds + position_embeddings

    print(hf_y.shape, pt_y.shape)
    print("Text embeddings match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_encoder(hf_model, torch_model, device="cuda"):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_embeddings = torch.rand(1, inputs['attention_mask'].shape[1], 512).to(device)
    attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], input_embeddings.dtype).to(device=device)
    with torch.no_grad():
        hf_y = hf_model.owlv2.text_model.encoder.forward(input_embeddings, None, attention_mask, False, False, False)[0]
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model.encoder.forward(input_embeddings, attention_mask)
    
    print(f"Text encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_embeddings_encoder(hf_model, torch_model, device="cuda"):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden = hf_model.owlv2.text_model.embeddings(inputs['input_ids'], None)
        attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], hidden.dtype)
        print(hidden.shape, attention_mask.shape)
        hf_y = hf_model.owlv2.text_model.encoder.forward(hidden, attention_mask, None, False, False, False)[0]
        hf_y = hf_model.owlv2.text_model.final_layer_norm(hf_y)
    with torch.no_grad():
        seq_length = inputs['input_ids'].shape[-1] if inputs['input_ids'] is not None else inputs_embeds.shape[-2]
        position_ids = torch_model.text_model.position_ids[:, :seq_length]
        inputs_embeds = torch_model.text_model.token_embedding(inputs['input_ids'])
        position_embeddings = torch_model.text_model.position_embedding(position_ids)
        hidden = inputs_embeds + position_embeddings
        attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], hidden.dtype)
        pt_y = torch_model.text_model.encoder.forward(hidden, attention_mask)
        pt_y = torch_model.text_model.final_layer_norm(pt_y)

    print(hf_y.shape, pt_y.shape)
    print(hf_y, )
    print(pt_y)
    print("Text embedding encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_tower(hf_model, torch_model, device="cuda"):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hf_y = hf_model.owlv2.text_model(inputs['input_ids'], inputs['attention_mask'], None, False, False, False)
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model(inputs['input_ids'], inputs['attention_mask'])
    print(f"Text tower match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_vision_tower(hf_model, torch_model, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model(inputs['pixel_values'])
        print(hf_y)
        y_cls,y_pool  = hf_y.last_hidden_state, hf_y.pooler_output
    with torch.no_grad():
        pt_y = torch_model.vision_model(inputs['pixel_values'])
        y_cls,y_pool = pt_y
    print(f"Vision tower match cls_token:", torch.allclose(y_cls, pt_y[0], atol=1e-3))
    print(f"Vision tower match full_map:", torch.allclose(y_pool, pt_y[1], atol=1e-3))

def test_obj_detection(hf_model, torch_model, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], text=["a cat"], return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        hf_y = hf_model(**inputs)
        hf_logits = hf_y.logits
        hf_pred_boxes = hf_y.pred_boxes
        hf_class_embeds = hf_y.class_embeds
        hf_objectness_logits = hf_y.objectness_logits

        hf_vision_feat = hf_y.image_embeds
        hf_text_raw = hf_y.text_embeds

        print(hf_vision_feat.shape, hf_text_raw.shape)

    
    with torch.no_grad():
        pt_pred_logits, pt_objectness_logits, pt_pred_boxes, pt_class_embeds, (vision_feat, text_raw) = torch_model.forward_object_detection(inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'])
        print(vision_feat.shape, text_raw.shape)

    print(hf_logits.shape, pt_pred_logits.shape)
    print(hf_pred_boxes.shape, pt_pred_boxes.shape)

    print(f"Vision feature match:", torch.allclose(hf_vision_feat, vision_feat, atol=1e-3))
    print(f"Text feature match:", torch.allclose(hf_text_raw, text_raw, atol=1e-3))

    print(f"Object detection logits match:", torch.allclose(hf_logits, pt_pred_logits, atol=1e-2))
    print(f"Object detection boxes match:", torch.allclose(hf_pred_boxes, pt_pred_boxes, atol=1e-3))
    print(f"Object detection class embeddings match:", torch.allclose(hf_class_embeds, pt_class_embeds, atol=1e-3))
    print(f"Object detection objectness logits match:", torch.allclose(hf_objectness_logits, pt_objectness_logits, atol=1e-2))

if __name__ == '__main__':
    hf_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    hf_model = hf_model.eval()
    
    torch_model = OwlV2()
    state_dict = {}
    with safe_open("model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens
    torch_model = torch_model.eval()
    torch_model.load_state_dict(state_dict, strict=True)
    
    torch_model.eval()
    hf_model.eval()

    #torch_model.cuda()
    #hf_model.cuda()

    weights = load_owlv2_weights(state_dict)
    
    for layer_idx in range(12):
        ...
        #test_attention(torch_model, weights, OWLV2_B16, device="cpu", layer_idx=layer_idx)
        #test_attention_masked(torch_model, weights, OWLV2_B16, device="cpu", layer_idx=layer_idx)
        #test_vision_encoder_layer(torch_model, weights, model_params=OWLV2_B16, device="cpu", layer_idx=layer_idx)
        #test_text_encoder_layer(hf_model, weights, device="cpu", layer_idx=i)
    #test_vision_encoder_full(torch_model, weights, model_params=OWLV2_B16, device="cpu")
    test_text_encoder_full(torch_model, weights, model_params=OWLV2_B16, device="cpu")
    quit()
    test_text_embeddings(hf_model, torch_model, device="cuda")
    test_text_encoder(hf_model, torch_model, device="cuda")
    test_text_embeddings_encoder(hf_model, torch_model, device="cuda")
    test_text_tower(hf_model, torch_model, device="cuda")
    test_vision_encoder(hf_model, torch_model, device="cuda")
    test_vision_tower(hf_model, torch_model, device="cuda")
    test_obj_detection(hf_model, torch_model, device="cuda")