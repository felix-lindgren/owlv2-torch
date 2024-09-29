import torch
import numpy as np
import utils
from PIL import Image
from hf_version.modeling_owlv2 import Owlv2ForObjectDetection
from hf_version.processing_owlv2 import Owlv2Processor
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from torch_version.owlv2 import OwlV2
from safetensors import safe_open


def test_vision_encoder_layer(hf_model, torch_model, layer_idx=0):
    x = torch.randn(1, 196, 768)
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model.encoder.layers[layer_idx].forward(x, None, None)[0]

    with torch.no_grad():
        pt_y = torch_model.vision_model.encoder.layers[layer_idx].forward(x)

    print(f"Vision encoder layer{layer_idx} match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_vision_encoder(hf_model, torch_model):
    x = torch.randn(1, 196, 768)
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model.encoder.forward(x, None, None, False, False, False)[0]

    with torch.no_grad():
        pt_y = torch_model.vision_model.encoder.forward(x)

    print(f"Vision encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_encoder_layer(hf_model, torch_model, layer_idx=0):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    input_embeddings = torch.rand(1, inputs['attention_mask'].shape[1], 512)
    with torch.no_grad():
        attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], input_embeddings.dtype)
        hf_y = hf_model.owlv2.text_model.encoder.layers[layer_idx].forward(input_embeddings, None, attention_mask)[0]
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model.encoder.layers[layer_idx].forward(input_embeddings, attention_mask)
    
    print(f"Text encoder layer{layer_idx} match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_embeddings(hf_model, torch_model):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
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

def test_text_encoder(hf_model, torch_model):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    input_embeddings = torch.rand(1, inputs['attention_mask'].shape[1], 512)
    attention_mask = _prepare_4d_attention_mask(inputs['attention_mask'], input_embeddings.dtype)
    with torch.no_grad():
        hf_y = hf_model.owlv2.text_model.encoder.forward(input_embeddings, None, attention_mask, False, False, False)[0]
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model.encoder.forward(input_embeddings, attention_mask)
    
    print(f"Text encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_text_embeddings_encoder(hf_model, torch_model):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")

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

def test_text_tower(hf_model, torch_model):
    test_sentence = "a cat"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(text=[test_sentence], return_tensors="pt")
    with torch.no_grad():
        hf_y = hf_model.owlv2.text_model(inputs['input_ids'], inputs['attention_mask'], None, False, False, False)
        hf_y = hf_y[0]
    with torch.no_grad():
        pt_y = torch_model.text_model(inputs['input_ids'], inputs['attention_mask'])
    print(f"Text tower match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_vision_tower(hf_model, torch_model):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], return_tensors="pt")
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model(inputs['pixel_values'])
        print(hf_y)
        y_cls,y_pool  = hf_y.last_hidden_state, hf_y.pooler_output
    with torch.no_grad():
        pt_y = torch_model.vision_model(inputs['pixel_values'])
        y_cls,y_pool = pt_y
    print(f"Vision tower match cls_token:", torch.allclose(y_cls, pt_y[0], atol=1e-3))
    print(f"Vision tower match full_map:", torch.allclose(y_pool, pt_y[1], atol=1e-3))
    

if __name__ == '__main__':
    hf_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    hf_model = hf_model.eval()
    
    torch_model = OwlV2()
    state_dict = {}
    with safe_open("weights/model.safetensors", framework="pt") as f:
        for k in f.keys():
            tens = f.get_tensor(k)
            k = k.replace("owlv2.","").replace(".embeddings","")
            k = k.replace("mlp.fc1","mlp.0").replace("mlp.fc2","mlp.2")
            state_dict[k] = tens
    torch_model = torch_model.eval()
    torch_model.load_state_dict(state_dict, strict=False)
    
    torch_model.eval()
    hf_model.eval()
    
    #for i in range(12):
    #    test_vision_encoder_layer(hf_model, torch_model, layer_idx=i)
        #test_text_encoder_layer(hf_model, torch_model, layer_idx=i)
    #test_text_embeddings(hf_model, torch_model)
    #test_text_encoder(hf_model, torch_model)
    #test_text_embeddings_encoder(hf_model, torch_model)
    #test_text_tower(hf_model, torch_model)
    #test_vision_encoder(hf_model, torch_model)
    test_vision_tower(hf_model, torch_model)