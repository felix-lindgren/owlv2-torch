import torch
from PIL import Image
from OWLv2torch.hf_version.modeling_owlv2 import Owlv2ForObjectDetection
from OWLv2torch.hf_version.processing_owlv2 import Owlv2Processor
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.utils.tokenizer import tokenize
from safetensors import safe_open


def test_vision_encoder_layer(hf_model, torch_model, device="cuda", layer_idx=0):
    x = torch.randn(1, 196, 768).to(device)
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model.encoder.layers[layer_idx].forward(x, None, None)[0]

    with torch.no_grad():
        pt_y = torch_model.vision_model.encoder.layers[layer_idx].forward(x)

    print(f"Vision encoder layer{layer_idx} match:", torch.allclose(hf_y, pt_y, atol=1e-4))

def test_vision_encoder(hf_model, torch_model, device="cuda"):
    x = torch.randn(1, 196, 768).to(device)
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model.encoder.forward(x, None, None, False, False, False)[0]

    with torch.no_grad():
        pt_y = torch_model.vision_model.encoder.forward(x)

    print(f"Vision encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4), torch.sub(hf_y, pt_y).max())

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
    
    print(f"Text encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4), torch.sub(hf_y, pt_y).max())

def test_vision_embedding(hf_model, torch_model, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pixel_values = inputs["pixel_values"]
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        patch_embeds = hf_model.owlv2.vision_model.embeddings.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = hf_model.owlv2.vision_model.embeddings.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        hf_y = embeddings + hf_model.owlv2.vision_model.embeddings.position_embedding(hf_model.owlv2.vision_model.embeddings.position_ids)
        hf_y = hf_model.owlv2.vision_model.pre_layernorm(hf_y)
        hf_y = hf_model.owlv2.vision_model.encoder(
            inputs_embeds=hf_y,
            output_hidden_states=False,
            return_dict=False,
            output_attentions=False
        )
        hf_y = hf_y[0]
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        patch_embeds = torch_model.vision_model.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        _class_embeds = torch_model.vision_model.class_embedding.expand(batch_size, 1, -1)
        _embeddings = torch.cat([_class_embeds, patch_embeds], dim=1)
        pt_y = _embeddings + torch_model.vision_model.position_embedding(torch_model.vision_model.position_ids)
        pt_y = torch_model.vision_model.pre_layernorm(pt_y)
        pt_y = torch_model.vision_model.encoder(pt_y, None)

    print(hf_y.shape, pt_y.shape)
    print(hf_y, )
    print(pt_y)
    print("Image embedding encoder match:", torch.allclose(hf_y, pt_y, atol=1e-4), torch.sub(hf_y, pt_y).max() )
 
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
        hf_y = hf_y[1]
    with torch.no_grad():
        token_ids = tokenize([test_sentence], context_length=16, truncate=True)
        attention_mask = token_ids == 0
        pt_y = torch_model.text_model(token_ids, attention_mask)
        pt_y = pt_y[0]
    print(f"Text tower match:", torch.allclose(hf_y, pt_y, atol=1e-4), torch.sub(hf_y, pt_y).max())

def test_vision_tower(hf_model, torch_model, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    #inputs["pixel_values"] = torch.rand((1,3,960,960))
    with torch.no_grad():
        hf_y = hf_model.owlv2.vision_model(inputs['pixel_values'])
        y_hidden_states,y_pool  = hf_y.last_hidden_state, hf_y.pooler_output
    with torch.no_grad():
        pt_y = torch_model.vision_model(inputs['pixel_values'])
        pt_pool, pt_full = pt_y
    print(f"Vision tower match full:", torch.allclose(y_hidden_states, pt_full, atol=1e-5), torch.sub(y_hidden_states, pt_full).max())
    print(f"Vision tower match pooled:", torch.allclose(y_pool, pt_pool, atol=1e-5), torch.sub(y_pool, pt_pool).max())

def test_obj_detection(hf_model, torch_model, device="cuda"):
    test_img = Image.open('img.jpg')
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    inputs = processor(images=[test_img], text=["a cat"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        hf_y = hf_model(**inputs)
        hf_logits = hf_y.logits
        hf_pred_boxes = hf_y.pred_boxes
        hf_class_embeds = hf_y.class_embeds
        hf_objectness_logits = hf_y.objectness_logits

        hf_vision_feat = hf_y.image_embeds
        hf_text_raw = hf_y.text_embeds
    
    with torch.no_grad():
        pt_pred_logits, pt_objectness_logits, pt_pred_boxes, pt_class_embeds, (vision_feat, text_raw) = torch_model.forward_object_detection(inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'])

    atol = 1e-4
    print(f"Vision feature match:", torch.allclose(hf_vision_feat, vision_feat, atol=atol), torch.sub(hf_vision_feat, vision_feat).max())
    print(f"Text feature match:", torch.allclose(hf_text_raw, text_raw, atol=atol), torch.sub(hf_text_raw, text_raw).max())

    print(f"Object detection logits match:", torch.allclose(hf_logits, pt_pred_logits, atol=atol), torch.sub(hf_logits, pt_pred_logits).max())
    print(f"Object detection boxes match:", torch.allclose(hf_pred_boxes, pt_pred_boxes, atol=atol), torch.sub(hf_pred_boxes, pt_pred_boxes).max())
    print(f"Object detection class embeddings match:", torch.allclose(hf_class_embeds, pt_class_embeds, atol=atol), torch.sub(hf_class_embeds, pt_class_embeds).max())
    print(f"Object detection objectness logits match:", torch.allclose(hf_objectness_logits, pt_objectness_logits, atol=atol), torch.sub(hf_objectness_logits, pt_objectness_logits).max())

if __name__ == '__main__':
    device = "cpu"
    hf_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    hf_model = hf_model.eval()
    
    torch_model = OwlV2()
    
    torch_model = torch_model.eval()
    torch_model.load_model("google/owlv2-base-patch16-ensemble")
    
    torch_model.eval()
    hf_model.eval()

    torch_model.to(device)
    hf_model.to(device)

    
    
    """ for i in range(12):
        test_vision_encoder_layer(hf_model, torch_model, device=device, layer_idx=i)
        test_text_encoder_layer(hf_model, torch_model, device=device, layer_idx=i)
    test_text_embeddings(hf_model, torch_model, device=device) """

    test_text_encoder(hf_model, torch_model, device=device)
    #test_text_embeddings_encoder(hf_model, torch_model, device=device)
    #test_text_tower(hf_model, torch_model, device=device)
    test_vision_embedding(hf_model, torch_model, device=device)
    #test_vision_encoder(hf_model, torch_model, device=device)
    #test_vision_tower(hf_model, torch_model, device=device)
    #test_obj_detection(hf_model, torch_model, device=device)