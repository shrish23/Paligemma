from PIL import Image
import torch
import fire
import requests
from io import BytesIO
from processing_paligemma import PaliGemmaProcessor
from modelling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not in the top-p
    probs_sort[mask] = 0.0
    # redistribute the probabilities so that they sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top-p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token




def get_model_inputs(processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str):

    # Check if image_file_path is a URL
    if image_file_path.startswith("http://") or image_file_path.startswith("https://"):
        response = requests.get(image_file_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file_path)

    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        device: str,
        prompt: str,
        image_file_path: str,
        max_tokens_to_generate: int,
        temperature: float,
        top_p: float,
        do_sample: bool
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]# Image tokens and text tokens
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )# Prefilling the kv_cache with the previous key and value tensors
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature(before softmax the logits are not probabilities, so we divide by temperature so as to reduce the effect of the logits, basically adding randomness)
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0) # Remove the batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token is generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input_ids
        input_ids = next_token.unsqueeze(-1)# We give the model the previously generated token as input for the next token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1,1), device = input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def main(
        model_path: str = None,
        prompt: str = None,
        image_file_path:str = None,
        max_tokens_to_generate: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = False,
        only_cpu: bool = False
    ):

    device = "cpu"

    # if not only_cpu:
    #     if torch.cuda.is_available():
    #         print("Using GPU")
    #         device = "cuda"
    #     elif torch.backends.mps.is_available():
    #         device = "mps"

    print("Device in use: ", device)

    print(f"Loading Model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running Inference.....")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample
        )

if __name__ == "__main__":
    fire.Fire(main)