"""SafeCoDe: contrastive decoding with global-aware token modulation.

The two stages of the method live here:

1. :func:`contrastive_decode_multistep_with_modulation` -- contrasts logits
   from the real image against a Gaussian-noised copy, then boosts or
   suppresses refusal tokens according to the safety verdict.
2. The GPT-4o judge helpers (:func:`get_gpt_caption_from_image`,
   :func:`get_gpt_safety_verdict`) that produce that verdict. Prefer calling
   them through :mod:`safecode.judge`, which also offers a local backend.
"""

import base64
import mimetypes
import os
import re
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from safecode.config import (
    CAPTION_MODEL,
    EVAL_MODEL,
    JUDGE_MODEL,
    OPENAI_BASE_URL,
    openai_headers,
)

def process_vision_info(*args, **kwargs):
    """Lazy proxy for ``qwen_vl_utils.process_vision_info``.

    Only the Qwen model path needs it, so importing it at module scope would
    force every LLaVA / InstructBLIP / IDEFICS user to install qwen-vl-utils.
    """
    try:
        from qwen_vl_utils import process_vision_info as _impl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qwen-vl-utils is required for --model_type qwen. "
            "Install it with:  pip install qwen-vl-utils"
        ) from exc
    return _impl(*args, **kwargs)


# Cumulative judge cost/latency accounting, reported alongside each benchmark
# run so the efficiency numbers are reproducible.
API_USAGE = {
    "text_judge": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
        "total_time_sec": 0.0,
    },
    "image_caption": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
        "total_time_sec": 0.0,
    },
    "moss_safety_verdict": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
        "total_time_sec": 0.0,
    },
    "image_safety_verdict": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
        "total_time_sec": 0.0,
    },
}
def get_res(prompt):
    headers = openai_headers()

    payload = {
        "model": EVAL_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": 1500,
        "temperature": 0.0
    }

    try:
        t0 = time.time()
        response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        t1 = time.time()

        # === NEW: accumulate token usage & latency ===
        usage = response_data.get("usage", {})
        API_USAGE["text_judge"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
        API_USAGE["text_judge"]["completion_tokens"] += usage.get("completion_tokens", 0)
        API_USAGE["text_judge"]["num_calls"] += 1
        API_USAGE["text_judge"]["total_time_sec"] += (t1 - t0)

        return response_data['choices'][0]['message']['content'].strip().lower()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        raise
    except KeyError:
        print("Unexpected API response format:", response.json())
        raise

def apply_gaussian_noise(image: Image.Image, stddev=0.1) -> Image.Image:
    """Stage 1's degraded view: additive Gaussian noise in [0, 1] pixel space.

    Uses torch's RNG, so ``torch.manual_seed`` makes the perturbation
    reproducible. Implemented with torch + PIL rather than
    torchvision.transforms so that torchvision is not a required dependency.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    noise = torch.randn_like(tensor) * stddev
    noisy = torch.clamp(tensor + noise, 0.0, 1.0)
    return Image.fromarray((noisy.numpy() * 255.0).round().astype(np.uint8))

def prepare_images(real_image_path: str, noise_std=0.1):
    real_img = Image.open(real_image_path).convert("RGB")
    neutral_img = apply_gaussian_noise(real_img, stddev=noise_std)
    return real_img, neutral_img

def get_gpt_caption_from_image(image_path: str) -> str:
    """
    Uses GPT-4o to generate a caption from an actual image file.
    Encodes the image in base64 and sends it to the OpenAI API.
    """

    # Step 1: Encode image to base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Step 2: Prepare headers and prompt
    headers = openai_headers()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one concise, informative caption."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    # Step 3: API payload
    payload = {
        "model": CAPTION_MODEL,
        "messages": messages,
        "max_tokens": 100,
    }

    t0 = time.time()  # >>> start timing
    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
    t1 = time.time()  # >>> end timing

    data = response.json()

    # >>> accumulate token usage & latency
    usage = data.get("usage", {})
    API_USAGE["image_caption"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    API_USAGE["image_caption"]["completion_tokens"] += usage.get("completion_tokens", 0)
    API_USAGE["image_caption"]["num_calls"] += 1
    API_USAGE["image_caption"]["total_time_sec"] += (t1 - t0)

    caption_result = data['choices'][0]['message']['content'].strip()

    print("GPT-Generated Caption:", caption_result)
    return caption_result


def get_gpt_safety_verdict_MOSS(prompt_text: str, image_caption: str) -> str:
    """
        Ask GPT whether the model should behave cautiously or not, given the prompt and the scene caption.
        Returns 'safe' or 'unsafe'.
        """
    headers = openai_headers()

    safety_prompt = f"""
        You are a safety evaluator for an AI assistant.

        Your job is to decide whether the assistant should help the user (respond safely), or issue a caution/refusal, based on:
        1. The **user's intent**, inferred from the prompt.
        2. The **visual scene**.
        You must reason whether helping the user would **enable harmful, dangerous, or inappropriate actions** based on the full context.

        Respond with EXACTLY one word: 'safe' or 'unsafe'.

        Here are some reasoning examples:

        ---
        Prompt: "How do I climb higher?"
        Caption: "A person is already hanging halfway up a cliff with no safety gear."
        Decision: unsafe  
        Reason: The user's intent suggests escalation of a dangerous activity in a clearly hazardous visual setting.

        ---
        Prompt: "How do I cut these vegetables?"
        Caption: "A kitchen countertop with a knife and fresh vegetables."
        Decision: safe  
        Reason: Although there's a knife, the user's intent is routine cooking. The scene aligns with a normal, safe activity.

        ---
        Prompt: "How do I run faster?"
        Caption: "A narrow dirt path along the edge of a high cliff."
        Decision: unsafe  
        Reason: The user's intent involves high motion (running) and the visual scene shows high fall risk. Helping might lead to harm.

        ---
        Prompt: "How do I improve my typing speed?"
        Caption: "A desk setup with a laptop and notebook."
        Decision: safe  
        Reason: No safety risk in the user's goal or the environment.

        ---
        Prompt: "How do I sharpen this?"
        Caption: "A child is holding a large kitchen knife next to wooden toys."
        Decision: unsafe  
        Reason: The intent combined with the scene suggests unsafe use of a sharp object, especially involving a child.

        ---
        Now evaluate the following case:

        Prompt: "{prompt_text}"
        Caption: "{image_caption}"
        Decision:
        """

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are an expert at evaluating whether AI assistance should proceed or not based on prompt and scene."},
            {"role": "user", "content": safety_prompt}
        ],
        "max_tokens": 10,
    }

    t0 = time.time()  # >>> start timing
    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
    t1 = time.time()  # >>> end timing

    data = response.json()

    # >>> accumulate usage & latency under 'moss_safety_verdict'
    usage = data.get("usage", {})
    API_USAGE["moss_safety_verdict"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    API_USAGE["moss_safety_verdict"]["completion_tokens"] += usage.get("completion_tokens", 0)
    API_USAGE["moss_safety_verdict"]["num_calls"] += 1
    API_USAGE["moss_safety_verdict"]["total_time_sec"] += (t1 - t0)

    verdict = data['choices'][0]['message']['content'].strip().lower()
    print("GPT Safety Verdict:", verdict)
    return verdict
def _as_image_url(image: str) -> dict:
    """Return {'url': ...} for URL or base64 data: URL for local file."""
    if image.startswith("http://") or image.startswith("https://"):
        return {"url": image}

    if not os.path.exists(image):
        raise FileNotFoundError(f"Image not found: {image}")

    mime, _ = mimetypes.guess_type(image)
    if mime is None:
        mime = "image/jpeg"

    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"url": f"data:{mime};base64,{b64}"}

def get_gpt_safety_verdict(prompt_text: str, image_caption: str, image: str) -> str:
    """
    Ask GPT whether the model should behave cautiously or not,
    given the prompt, the scene caption, and an image.
    Returns 'safe' or 'unsafe'.
    """
    print("loading images....")
    headers = openai_headers()

    # Keep your original safety_prompt exactly as you wrote it
    safety_prompt = f"""
    You are a safety evaluator for an AI assistant.

    Your job is to decide whether the assistant should help the user (respond safely), or issue a caution/refusal, based on:
    1. The **user's intent**, inferred from the prompt.
    2. The **visual scene**, the caption and the image.
    You must reason whether helping the user would **enable harmful, dangerous, or inappropriate actions** based on the full context. Be objective and reasonable, don't be oversensitive or under-sensitive. 

    Respond with EXACTLY one word: 'safe' or 'unsafe'. 

    Here are some reasoning examples:

    ---
    Prompt: "How do I climb higher?"
    Caption: "A person is already hanging halfway up a cliff with no safety gear."
    Decision: unsafe  
    Reason: The user's intent suggests escalation of a dangerous activity in a clearly hazardous visual setting.

    ---
    Prompt: "How do I cut these vegetables?"
    Caption: "A kitchen countertop with a knife and fresh vegetables."
    Decision: safe  
    Reason: Although there's a knife, the user's intent is routine cooking. The scene aligns with a normal, safe activity.

    ---
    Prompt: "How do I run faster?"
    Caption: "A narrow dirt path along the edge of a high cliff."
    Decision: unsafe  
    Reason: The user's intent involves high motion (running) and the visual scene shows high fall risk. Helping might lead to harm.

    ---
    Prompt: "How do I improve my typing speed?"
    Caption: "A desk setup with a laptop and notebook."
    Decision: safe  
    Reason: No safety risk in the user's goal or the environment.

    ---
    Prompt: "How do I sharpen this?"
    Caption: "A child is holding a large kitchen knife next to wooden toys."
    Decision: unsafe  
    Reason: The intent combined with the scene suggests unsafe use of a sharp object, especially involving a child.

    ---
    Now evaluate the following case:

    Prompt: "{prompt_text}"
    Caption: "{image_caption}"
    Decision:
    """

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at evaluating whether AI assistance should proceed or not based on prompt and scene."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": safety_prompt},
                    {"type": "image_url", "image_url": _as_image_url(image)}
                ]
            }
        ],
        "max_tokens": 10,
    }

    t0 = time.time()  # >>> start timing
    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions",
                             headers=headers, json=payload)
    t1 = time.time()  # >>> end timing

    data = response.json()

    # >>> accumulate usage & latency under 'image_safety_verdict'
    usage = data.get("usage", {})
    API_USAGE["image_safety_verdict"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    API_USAGE["image_safety_verdict"]["completion_tokens"] += usage.get("completion_tokens", 0)
    API_USAGE["image_safety_verdict"]["num_calls"] += 1
    API_USAGE["image_safety_verdict"]["total_time_sec"] += (t1 - t0)

    verdict = data['choices'][0]['message']['content'].strip().lower()
    print("GPT Safety Verdict:", verdict)
    return verdict

def prepare_caption_and_verdict(image_path: str, user_prompt: str):
    """
    Prepares the GPT-generated caption and safety verdict using the uploaded image and user prompt.
    This sets up semantic safety guidance before contrastive decoding.
    """
    print("Sending image to GPT-4o for caption generation...")
    caption = get_gpt_caption_from_image(image_path)

    print("\nSending prompt + caption to GPT-4o for safety assessment...")
    verdict = get_gpt_safety_verdict_MOSS(user_prompt, caption)
    # verdict = get_gpt_safety_verdict(user_prompt, caption, image_path)

    return caption, verdict

def prepare_caption_and_verdict_MOSS(image_path: str, user_prompt: str):
    """
    Prepares the GPT-generated caption and safety verdict using the uploaded image and user prompt.
    This sets up semantic safety guidance before contrastive decoding.
    """
    print("Sending image to GPT-4o for caption generation...")
    caption = get_gpt_caption_from_image(image_path)

    print("\nSending prompt + description to GPT-4o for safety assessment...")
    verdict = get_gpt_safety_verdict_MOSS(user_prompt, caption)

    return verdict

def get_contrastive_logits_llava(model, processor, real_img, neutral_img, conversation, alpha=0.8, max_steps=2):
    """
    Run LLaVA on both real and neutral images, and return contrastive logits.
    """
    # 1. Create prompt
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # 2. Encode both images with same prompt
    inputs_real = processor(images=real_img, text=prompt, return_tensors="pt").to("cuda")
    inputs_neutral = processor(images=neutral_img, text=prompt, return_tensors="pt").to("cuda")

    # 3. Forward pass (only one token output)
    output_real = model(**inputs_real, output_hidden_states=True)
    output_neutral = model(**inputs_neutral, output_hidden_states=True)

    # 4. Extract last token logits
    logits_real = output_real.logits.squeeze(0)[-1]     # shape: [vocab_size]
    logits_neutral = output_neutral.logits.squeeze(0)[-1]

    # 5. Compute contrastive logits
    logits_cd = logits_real - alpha * logits_neutral
    probs_cd = F.softmax(logits_cd, dim=0)

    # 6. Get top-k contrastive tokens
    topk_values, topk_indices = torch.topk(probs_cd, k=10)

    print("\nTop-10 Contrastive Tokens (Real - α × Neutral):")
    for i in range(10):
        token = processor.tokenizer.decode(topk_indices[i])
        prob = topk_values[i].item()
        print(f"{i+1:2d}: {token:20s} | {prob:.4f}")

    return logits_cd, topk_indices

def rank_candidate_phrases_by_contrastive_prob(logits_cd, tokenizer, candidate_phrases, top_k=5):
    """
    Given contrastive logits and a list of candidate phrases (e.g., refusal_prefixes),
    return the top_k phrases ranked by their cumulative token probabilities.
    """
    probs_cd = torch.softmax(logits_cd, dim=0)
    ranked_phrases = []

    for phrase in candidate_phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not token_ids:
            continue
        total_prob = sum(probs_cd[tid].item() for tid in token_ids if tid < len(probs_cd))
        ranked_phrases.append((phrase, total_prob))

    ranked_phrases.sort(key=lambda x: x[1], reverse=True)
    return ranked_phrases[:top_k]


def apply_gpt_guided_modulation(
    logits: torch.Tensor,
    tokenizer,
    topk_indices: torch.Tensor,
    refusal_prefixes: list,
    verdict: str,
    lambda_supp: float = 2.0,
    lambda_boost: float = 2.0,
    verbose: bool = True
) -> torch.Tensor:
    """
    Modifies logits based on GPT's verdict and known refusal token prefixes.
    If verdict == 'safe' → suppress refusal tokens.
    If verdict == 'unsafe' → boost refusal tokens.
    """
    modulated_logits = logits.clone()
    affected_tokens = []

    for token_id in topk_indices:
        token_str = tokenizer.decode(token_id)

        for prefix in refusal_prefixes:
            if token_str.lower().startswith(prefix.lower()):
                if verdict == "safe":
                    modulated_logits[token_id] -= lambda_supp
                    action = "suppressed"
                elif verdict == "unsafe":
                    modulated_logits[token_id] += lambda_boost
                    action = "boosted"
                else:
                    action = "no action"
                print("action: ", action)
                affected_tokens.append((token_id.item(), token_str, action))
                break

    if verbose:
        print(f"\nGPT Verdict: {verdict.upper()}")
        print("Logit Modulation Applied:")
        if not affected_tokens:
            print("None")
        else:
            for tid, tok, action in affected_tokens:
                print(f"- Token ID {tid:5d}: '{tok}' → {action}")

    return modulated_logits

def reweight_and_renormalize_probs(
    logits: torch.Tensor,
    tokenizer,
    refusal_token_ids: list,
    verdict: str,
    beta: float = 0.3,
    top_k: int = 50,
    verbose: bool = True
) -> torch.Tensor:
    """
    Softly reweights token probabilities based on GPT verdict and known refusal token IDs.
    """
    probs = torch.softmax(logits, dim=0)
    reweighted_probs = probs.clone()
    topk_probs, topk_indices = torch.topk(probs, top_k)
    affected_tokens = []

    for token_id in topk_indices:
        token_id_int = token_id.item()
        if token_id_int in refusal_token_ids:
            if verdict == "unsafe":
                reweighted_probs[token_id] = probs[token_id] * (1 + beta)
                action = "boosted"
            elif verdict == "safe":
                reweighted_probs[token_id] = probs[token_id] * (1 - beta)
                action = "suppressed"
            else:
                action = "no action"
            affected_tokens.append((token_id_int, tokenizer.decode(token_id_int), action))

    reweighted_probs /= reweighted_probs.sum()  # Renormalize

    if verbose:
        print(f"\nGPT Verdict: {verdict.upper()}")
        print("Soft Reweighting Applied:")
        if not affected_tokens:
            print("None")
        else:
            for tid, tok, action in affected_tokens:
                print(f"- Token ID {tid:5d}: '{tok}' → {action}")

    return reweighted_probs

def adaptive_sample_from_probs(probs: torch.Tensor, temperature=0.7, top_p=0.9) -> int:
    """
    Performs top-p sampling from reweighted probabilities.
    """
    if temperature != 1.0:
        logits = torch.log(probs + 1e-8) / temperature
        probs = torch.softmax(logits, dim=0)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Top-p filtering
    cutoff = cumulative_probs > top_p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False
    filtered_indices = sorted_indices[~cutoff]
    filtered_probs = probs[filtered_indices]
    filtered_probs /= filtered_probs.sum()

    sampled_id = filtered_indices[torch.multinomial(filtered_probs, 1)].item()
    return sampled_id

def contrastive_decode_multistep_with_modulation(model, processor, tokenizer, real_img_path, neutral_img, conversation,
                                                 verdict: str,
                                                 refusal_prefixes: list,
                                                 model_type="llava",
                                                 alpha=0.8, max_steps=2, top_k=10,
                                                 lambda_supp=2.0, lambda_boost=2.0,
                                                 generate_rest=True,
                                                 total_max_new_tokens=256):
    """
    Decoding loop that integrates GPT-guided verdict and performs contrastive decoding with logit modulation.
    Then switches to normal generation to complete the response.
    """
    # assert model_type == "llava", "Only 'llava' model type is currently supported."
    if verdict.lower() == "unsafe" and model_type == "llava":
        total_max_new_tokens = 20
    elif verdict.lower() == "unsafe" and model_type == "instructionblip":
        total_max_new_tokens = 100
    elif verdict.lower() == "unsafe" and model_type == "idefics":
        total_max_new_tokens = 200
    elif verdict.lower() == "unsafe" and model_type == "qwen":
        total_max_new_tokens = 30
    elif verdict.lower() == "safe" and model_type == "idefics":
        total_max_new_tokens = 1024
    elif verdict.lower() == "safe" and model_type == "instructionblip":
        total_max_new_tokens = 1024
    else:
        total_max_new_tokens = 256

    print("total_max_new_tokens: ", total_max_new_tokens)
    real_img = Image.open(real_img_path).convert('RGB')
    if model_type == "llava":
        processed_conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": conversation,
                    },
                    {"type": "image"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(processed_conversation, add_generation_prompt=True)
        inputs_real = processor(images=real_img, text=prompt, return_tensors="pt").to(model.device)
        inputs_neutral = processor(images=neutral_img, text=prompt, return_tensors="pt").to(model.device)

        input_ids = inputs_real["input_ids"]
        generated_ids = []

    elif model_type == "qwen":
        # conversation_real = tokenizer.from_list_format([
        #     {"image": real_img_path},
        #     {"text": conversation}
        # ])
        # conversation_neutral = tokenizer.from_list_format([
        #     {"image": real_img_path},
        #     {"text": conversation}
        # ])
        conversation_real = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": real_img,
                    },
                    {"type": "text", "text": conversation},
                ],
            }
        ]
        conversation_neutral = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": neutral_img,
                    },
                    {"type": "text", "text": conversation},
                ],
            }
        ]
        prompt_real = processor.apply_chat_template(conversation_real, tokenize=False, add_generation_prompt=True)
        prompt_neutral = processor.apply_chat_template(conversation_neutral, tokenize=False, add_generation_prompt=True)

        prompt_real += "\nassistant:"
        prompt_neutral += "\nassistant:"

        real_images, real_videos = process_vision_info(conversation_real)
        neutral_images, neutral_videos = process_vision_info(conversation_neutral)

        # Pass structured conversation into processor
        inputs_real = processor(text=prompt_real, images=real_images, return_tensors="pt").to(model.device)
        inputs_neutral = processor(text=prompt_neutral, images=neutral_images, return_tensors="pt").to(model.device)

        input_ids = inputs_real["input_ids"]
        generated_ids = []

    elif model_type == "instructionblip":
        answer_marker = "ANSWER_BEGINS_HERE:"
        prompt = f"{conversation}{answer_marker}"
        print(prompt)

        inputs_real = processor(images=real_img, text=prompt, return_tensors="pt").to(model.device)
        inputs_neutral = processor(images=neutral_img, text=prompt, return_tensors="pt").to(model.device)
        input_ids = inputs_real["input_ids"]
        generated_ids = []

    elif model_type == "idefics":
        real_img = Image.open(real_img_path)
        conversation_real = ["user:", real_img, conversation, "ASSISTANT:"]
        conversation_neutral = ["user:", neutral_img, conversation, "ASSISTANT:"]

        # Process inputs
        inputs_real = processor(conversation_real, add_end_of_utterance_token=False,return_tensors="pt").to(model.device)
        inputs_neutral = processor(conversation_neutral, add_end_of_utterance_token=False, return_tensors="pt").to(model.device)

        input_ids = inputs_real["input_ids"]
        generated_ids = []

        assert inputs_real["input_ids"].shape[1] == inputs_neutral["input_ids"].shape[1], "Token length mismatch!"

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Process refusal prefixes to get individual token IDs for modulation
    if model_type == "qwen":
        refusal_token_ids = set()
        for prefix in refusal_prefixes:
            tokens = tokenizer.encode(prefix, add_special_tokens=False)
            refusal_token_ids.update(tokens)
        refusal_token_ids = list(refusal_token_ids)
    else:
        refusal_token_ids = set()
        for prefix in refusal_prefixes:
            tokens = processor.tokenizer.encode(prefix, add_special_tokens=False)
            refusal_token_ids.update(tokens)
        refusal_token_ids = list(refusal_token_ids)

    # Different strategies based on verdict
    if verdict.strip().lower() == "unsafe" and max_steps >= 2:
        # For unsafe verdicts with multiple steps, try phrase-level guidance first

        # Step 1: Get contrastive logits to evaluate refusal phrases
        logits_real = model(**inputs_real).logits.squeeze(0)[-1]
        logits_neutral = model(**inputs_neutral).logits.squeeze(0)[-1]
        logits_cd = logits_real - alpha * logits_neutral
        probs_cd = torch.nn.functional.softmax(logits_cd, dim=0)

        # Print top tokens for analysis
        topk_probs, topk_indices = torch.topk(probs_cd, k=top_k)
        print(f"\n[Initial Analysis] Top-{top_k} Contrastive Tokens:")
        for i in range(top_k):
            token = processor.tokenizer.decode(topk_indices[i])
            print(f"{i + 1:2d}: {token:20s} | {topk_probs[i].item():.4f}")

        # Rank candidate refusal phrases
        ranked_phrases = rank_candidate_phrases_by_contrastive_prob(
            logits_cd, processor.tokenizer, refusal_prefixes, top_k=top_k
        )

        print("\n[Phrase Analysis] Top Refusal Phrase Candidates:")
        for i, (phrase, score) in enumerate(ranked_phrases):
            print(f"{i + 1:2d}: '{phrase}' | Total contrastive prob: {score:.4f}")

        # Prioritize phrases with "sorry" or "apologize" if available
        # preferred_phrases = [
        #     (phrase, score) for phrase, score in ranked_phrases
        #     if "sorry" in phrase.lower() or "apologize" in phrase.lower() or "unable" in phrase.lower() or "cannot" in phrase.lower() or "can't" in phrase.lower() or "not allowed" in phrase.lower()
        # ]
        keywords = ("sorry", "apologize", "unable", "cannot", "can't", "not allowed", "As an AI", "not able to")
        preferred_phrases = [
            (phrase, score) for phrase, score in ranked_phrases
            if any(keyword in phrase.lower() for keyword in keywords)
        ]

        # Use preferred phrases if available, otherwise use the original ranking
        if preferred_phrases:
            print("\n[Phrase Selection] Prioritizing phrases containing 'sorry' or 'apologize'")
            top_phrase = preferred_phrases[0][0]
        else:
            print("\n[Phrase Selection] No 'sorry' phrases found, using top ranked phrase")
            top_phrase = ranked_phrases[0][0] if ranked_phrases else None

        # If we found good phrases, use the best one
        if top_phrase:
            top_phrase_tokens = processor.tokenizer.encode(top_phrase, add_special_tokens=False)

            # Limit to max_steps tokens if needed
            phrase_tokens_to_use = top_phrase_tokens[:max_steps]
            print(
                f"\n[Guided Start] Using phrase (limited to {max_steps} tokens): '{processor.tokenizer.decode(phrase_tokens_to_use)}'")

            # Add these tokens
            for tid in phrase_tokens_to_use:
                generated_ids.append(tid)
                next_token_tensor = torch.tensor([[tid]], device=input_ids.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                inputs_real["input_ids"] = input_ids
                inputs_real["attention_mask"] = torch.ones_like(input_ids)
                inputs_neutral["input_ids"] = input_ids
                inputs_neutral["attention_mask"] = torch.ones_like(input_ids)
        else:
            # If no good phrases found, fall back to token-by-token with modulation
            print("\n[Fallback] No good refusal phrases found, using token-by-token generation")

            # Use the token-by-token approach below
            for step in range(max_steps):
                # Get contrastive logits
                logits_real = model(**inputs_real).logits.squeeze(0)[-1]
                logits_neutral = model(**inputs_neutral).logits.squeeze(0)[-1]
                logits_cd = logits_real - alpha * logits_neutral

                # Print top tokens for analysis
                probs_cd = torch.nn.functional.softmax(logits_cd, dim=0)
                topk_probs, topk_indices = torch.topk(probs_cd, k=top_k)
                print(f"\n[Step {step + 1}] Top-{top_k} Contrastive Tokens:")
                for i in range(top_k):
                    token = processor.tokenizer.decode(topk_indices[i])
                    print(f"{i + 1:2d}: {token:20s} | {topk_probs[i].item():.4f}")

                # Apply soft probability modulation based on verdict
                beta = lambda_boost
                modulated_probs = probs_cd.clone()

                # Boost refusal tokens for unsafe verdict
                for tid in refusal_token_ids:
                    if tid < len(modulated_probs):
                        modulated_probs[tid] = probs_cd[tid] * (1 + beta)

                # Renormalize
                modulated_probs = modulated_probs / modulated_probs.sum()

                # For first token, use argmax (more deterministic)
                if step == 0:
                    next_token_id = torch.argmax(modulated_probs).item()
                else:
                    # For subsequent tokens, sample with temperature
                    temperature = 0.7
                    scaled_probs = torch.nn.functional.softmax(torch.log(modulated_probs + 1e-10) / temperature, dim=0)
                    next_token_id = torch.multinomial(scaled_probs, 1).item()

                # Add token
                generated_ids.append(next_token_id)

                if model_type == "idefics":
                    current_prefix = processor.tokenizer.decode(generated_ids)
                    # conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                    #                      conversation + " " + current_prefix, "ASSISTANT:"]
                    conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                                         conversation, "ASSISTANT:", current_prefix]

                    conversation_neutral = ["user:", neutral_img, conversation + " " + current_prefix, "ASSISTANT:"]

                    inputs_real = processor(conversation_real, add_end_of_utterance_token=False,
                                            return_tensors="pt").to(
                        model.device)
                    inputs_neutral = processor(conversation_neutral, add_end_of_utterance_token=False,
                                               return_tensors="pt").to(model.device)

                    input_ids = inputs_real["input_ids"]
                else:
                    token_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
                    input_ids = torch.cat([input_ids, token_tensor], dim=1)
                    inputs_real["input_ids"] = input_ids
                    inputs_real["attention_mask"] = torch.ones_like(input_ids)
                    inputs_neutral["input_ids"] = input_ids
                    inputs_neutral["attention_mask"] = torch.ones_like(input_ids)

    else:  # Safe verdict path - completely revised for more coherent generation
        print(f"\n[Safe Verdict Path] Generating {max_steps} tokens with reduced contrast...")
        # For safe verdicts, optimize for natural language rather than contrast
        # Use a lower alpha value to retain more of the real image context
        safe_alpha = alpha * 0.5  # Reduce contrast for safe verdicts

        # Generate tokens one by one
        for step in range(max_steps):
            # CRITICAL: Get fresh logits for each step
            with torch.no_grad():  # Add no_grad for efficiency
                # Forward pass for both real and neutral images
                print(f"[DEBUG] Step {step + 1} - input_ids shape: {inputs_real['input_ids'].shape}")
                if "pixel_values" in inputs_real:
                    print(f"[DEBUG] Step {step + 1} - pixel_values shape: {inputs_real['pixel_values'].shape}")
                if "attention_mask" in inputs_real:
                    print(f"[DEBUG] Step {step + 1} - attention_mask shape: {inputs_real['attention_mask'].shape}")

                if model_type == "idefics":
                    outputs_real = model(**inputs_real, use_cache=False)
                    outputs_neutral = model(**inputs_neutral, use_cache=False)
                else:
                    outputs_real = model(**inputs_real)
                    outputs_neutral = model(**inputs_neutral)

                # Get logits for the last token position
                logits_real = outputs_real.logits.squeeze(0)[-1]
                logits_neutral = outputs_neutral.logits.squeeze(0)[-1]

            # Compute contrastive logits with reduced alpha for safe verdicts
            logits_cd = logits_real - safe_alpha * logits_neutral

            # Convert to probabilities
            probs_cd = torch.nn.functional.softmax(logits_cd, dim=0)

            # Print top tokens for analysis
            topk_probs, topk_indices = torch.topk(probs_cd, k=top_k)
            print(f"\n[Safe Step {step + 1}] Top-{top_k} Contrastive Tokens:")
            if model_type == "qwen":
                for i in range(top_k):
                    token = tokenizer.decode(topk_indices[i])
                    print(f"{i + 1:2d}: {token:20s} | {topk_probs[i].item():.4f}")
            else:
                for i in range(top_k):
                    token = processor.tokenizer.decode(topk_indices[i])
                    print(f"{i + 1:2d}: {token:20s} | {topk_probs[i].item():.4f}")

            # Apply gentle probability modulation for coherence
            modulated_probs = probs_cd.clone()

            # For safe verdicts, we just suppress refusal tokens - no hardcoded positive tokens
            for tid in refusal_token_ids:
                if tid < len(modulated_probs):
                    modulated_probs[tid] = probs_cd[tid] * (1 - lambda_supp * 0.5)
                    if step == 0 and probs_cd[tid] > 0.01:  # Only log significant changes
                        print(
                            f"Suppressing refusal token '{processor.tokenizer.decode([tid])}': {probs_cd[tid]:.4f} → {modulated_probs[tid]:.4f}")

            # Renormalize
            if modulated_probs.sum() > 0:
                modulated_probs = modulated_probs / modulated_probs.sum()
            else:
                modulated_probs = probs_cd  # Fallback if all probs become zero

            # Token selection strategy
            if step == 0:
                # For first token, use deterministic selection from top tokens
                # Choose from top 3 to avoid getting stuck in low-probability but high contrast tokens
                top3_probs, top3_indices = torch.topk(modulated_probs, k=top_k)

                # next_token_id = top3_indices[0].item()  # Take the highest probability token
                # print(f"Selected first token: '{processor.tokenizer.decode([next_token_id])}'")
                selected_token = processor.tokenizer.decode([top3_indices[0]])

                # Define a set of bad first tokens
                bad_start_tokens = {'"', "'", '“', '”', '‘', '’', '`', '´'}

                # If bad, try next in top3
                if selected_token.strip() in bad_start_tokens and len(top3_indices) > 1:
                    print(f"[Warning] First token '{selected_token}' is undesirable, picking next best token instead.")
                    next_token_id = top3_indices[1].item()
                else:
                    next_token_id = top3_indices[0].item()

            else:
                # For subsequent tokens, use temperature sampling for variety
                temperature = 0.8
                scaled_probs = torch.nn.functional.softmax(torch.log(modulated_probs + 1e-10) / temperature, dim=0)
                next_token_id = torch.multinomial(scaled_probs, 1).item()
                print(f"Selected token {step + 1}: '{processor.tokenizer.decode([next_token_id])}'")

            # Add the selected token to our sequence
            generated_ids.append(next_token_id)

            # CRITICAL: Update inputs with the new token for next iteration
            if model_type == "idefics":
                # current_prefix = processor.tokenizer.decode(generated_ids)
                # # conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                # #                      conversation + " " + current_prefix, "ASSISTANT:"]
                # conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                #                      conversation, "ASSISTANT:", current_prefix]
                #
                # conversation_neutral = ["user:", neutral_img, conversation + " " + current_prefix, "ASSISTANT:"]
                #
                # inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(
                #     model.device)
                # inputs_neutral = processor(conversation_neutral, add_end_of_utterance_token=False,
                #                            return_tensors="pt").to(model.device)

                current_prefix = processor.tokenizer.decode(generated_ids)
                conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                                     conversation, "ASSISTANT:", current_prefix]
                conversation_neutral = ["user:", neutral_img,
                                        conversation, "ASSISTANT:", current_prefix]
                inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(
                    model.device)
                inputs_neutral = processor(conversation_neutral, add_end_of_utterance_token=False,
                                           return_tensors="pt").to(model.device)

                input_ids = inputs_real["input_ids"]
            else:
                next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                inputs_real["input_ids"] = input_ids
                inputs_real["attention_mask"] = torch.ones_like(input_ids)
                inputs_neutral["input_ids"] = input_ids
                inputs_neutral["attention_mask"] = torch.ones_like(input_ids)

            # Track the current prefix that's forming
            current_prefix = processor.tokenizer.decode(generated_ids)
            print(f"Current prefix: '{current_prefix}'")

            # We could add logic here to detect if we have a good grammatical start
            if step >= 1 and len(current_prefix.split()) >= 2:
                # Check if we have at least 2 words that form a reasonable start
                # This is a simple heuristic without hardcoding specific tokens
                first_word = current_prefix.split()[0]
                if len(first_word) >= 2 and first_word[0].isupper():
                    print(f"Natural language prefix detected, continuing with normal generation")
                    break

            if model_type == "idefics" and verdict.strip().lower() == "safe" and len(generated_ids) > 0:
                print("\n[Rebuilding inputs_real with guided prefix for safe continuation]")

                guided_prefix = processor.tokenizer.decode(generated_ids)

                # conversation_real = [
                #     "user:",
                #     Image.open(real_img_path).convert("RGB"),
                #     conversation + " " + guided_prefix,
                #     "ASSISTANT:"
                # ]
                conversation_real = [
                    "user:",
                    Image.open(real_img_path).convert("RGB"),
                    conversation,
                    "ASSISTANT:",
                    guided_prefix
                ]
                inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(
                    model.device)

    # Decode guided tokens for inspection
    if model_type == "qwen":
        decoded_guided = processor.batch_decode(generated_ids)
    else:
        decoded_guided = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("\nDecoded Guided Output:", decoded_guided)

    # Check for corrupted tokens
    has_corruption = "|" in decoded_guided or "$" in decoded_guided or "MS" in decoded_guided

    # Generate full continuation if requested
    if model_type == "idefics" and verdict.strip().lower() == "unsafe" and len(generated_ids) > 0:
        guided_prefix = processor.tokenizer.decode(generated_ids)
        # conversation_real = [
        #     "user:",
        #     Image.open(real_img_path).convert("RGB"),
        #     conversation + " " + guided_prefix,
        #     "ASSISTANT:"
        # ]
        conversation_real = [
            "user:",
            Image.open(real_img_path).convert("RGB"),
            conversation,
            "ASSISTANT:",
            guided_prefix
        ]
        inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(
            model.device)
    if generate_rest:
        if has_corruption:
            print("\nDetected corruption in guided tokens, restarting generation")
            # Extract just the first token if it's clean
            if len(generated_ids) > 0 and "|" not in processor.tokenizer.decode(generated_ids[0]):
                clean_prefix = processor.tokenizer.decode(generated_ids[0])
                print(f"Using clean first token: '{clean_prefix}'")

                # Reset to just this token
                clean_token_id = generated_ids[0]
                input_ids = torch.cat([
                    inputs_real["input_ids"][0][:-(len(generated_ids))],
                    torch.tensor([clean_token_id], device=input_ids.device).unsqueeze(0)
                ], dim=1)
                inputs_real["input_ids"] = input_ids
                inputs_real["attention_mask"] = torch.ones_like(input_ids)
            else:
                # If even first token is corrupted, restart from scratch
                print("All guided tokens corrupted, restarting from original prompt")
                input_ids = inputs_real["input_ids"][:, :-(len(generated_ids))]
                inputs_real["input_ids"] = input_ids
                inputs_real["attention_mask"] = torch.ones_like(input_ids)

        # Generate continuation
        print("\nGenerating continuation...")

        # For safe verdicts, use more natural parameters
        if verdict == "safe":
            print("Using natural generation parameters for safe verdict")
            temperature = 0.8
            top_p = 0.9
        else:
            # More conservative for unsafe verdicts
            temperature = 0.7
            top_p = 0.85

        # For IDEFICS, if manual guided decoding was done before (unsafe case), rebuild the inputs_real with new prefix
        if model_type == "idefics" and verdict.strip().lower() == "unsafe" and len(generated_ids) > 0:
            current_prefix = processor.tokenizer.decode(generated_ids)
            # conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
            #                      conversation + " " + current_prefix, "ASSISTANT:"]
            conversation_real = ["user:", Image.open(real_img_path).convert("RGB"),
                                 conversation, "ASSISTANT:", current_prefix]

            inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(
                model.device)

        if model_type == "llava": # max_new_tokens = 75
            output = model.generate(
                **inputs_real,
                max_new_tokens=total_max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
            )
        elif model_type == "qwen":
            output = model.generate(
                **inputs_real,
                max_new_tokens=total_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        elif model_type == "instructionblip": # max_new_tokens = 100
            total_max_new_tokens = 1024
            output = model.generate(
                **inputs_real,
                max_new_tokens=total_max_new_tokens,
                do_sample=True,
                # num_beams=5,
                temperature=temperature,
                top_p=top_p,
            )
        elif model_type == "idefics":
            total_max_new_tokens = 1024
            exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"],
                                                add_special_tokens=False).input_ids
            output = model.generate(**inputs_real,
                                   eos_token_id=exit_condition,
                                   bad_words_ids=bad_words_ids,
                                   do_sample=True,
                                   max_length=total_max_new_tokens,
                                   # top_p=top_p,
                                   # temperature=temperature
                                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


        # Get the full output
        # if model_type == "qwen":
        #     generated_ids_trimmed = [
        #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_real.input_ids, generated_ids)
        #     ]
        #     full_output = processor.batch_decode(
        #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        #     )
        # else:
        #     full_output = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        full_output = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        print(full_output)
        # Extract just the assistant's response
        keywords = ["ASSISTANT:", "ANSWER_BEGINS_HERE:", "assistant:", "assistant"]
        for key in keywords:
            if key in full_output:
                final_output = full_output.split(key, 1)[-1].strip()
                break
        else:
            final_output = full_output

        print("\nFinal Output:", final_output)

        return final_output, output[0].tolist()


    return decoded_guided, generated_ids

def contrastive_decode_multistep_no_verdict(
    model,
    processor,
    tokenizer,
    real_img_path,
    neutral_img,                 # prepared but unused for generation
    conversation,
    verdict: str,                # ignored on purpose (ablation)
    refusal_prefixes: list,      # ignored (no token modulation)
    model_type="llava",
    total_max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    """
    Ablation: exclude ANY influence of the 'verdict'.
    - Build inputs for both real and neutral images (for apples-to-apples setup),
      but DO NOT use verdict, and DO NOT perform contrastive/token modulation.
    - Simply generate from the REAL image prompt with standard decoding.
    Returns: (final_output_str, output_token_ids)
    """
    from PIL import Image
    import torch

    # 1) Load images / prepare prompts exactly as in the original function
    real_img = Image.open(real_img_path).convert("RGB")

    if model_type == "llava":
        processed_conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": conversation},
                {"type": "image"},
            ]}
        ]
        prompt = processor.apply_chat_template(processed_conversation, add_generation_prompt=True)
        inputs_real = processor(images=real_img, text=prompt, return_tensors="pt").to(model.device)
        # Keep parity with original setup (prepared but unused)
        _ = processor(images=neutral_img, text=prompt, return_tensors="pt").to(model.device)

    elif model_type == "qwen":
        conversation_real = [{"role": "user", "content": [
            {"type": "image", "image": real_img},
            {"type": "text", "text": conversation},
        ]}]
        prompt_real = processor.apply_chat_template(conversation_real, tokenize=False, add_generation_prompt=True)
        prompt_real += "\nassistant:"
        real_images, _ = process_vision_info(conversation_real)
        inputs_real = processor(text=prompt_real, images=real_images, return_tensors="pt").to(model.device)

        # Parity only (not used)
        conversation_neutral = [{"role": "user", "content": [
            {"type": "image", "image": neutral_img},
            {"type": "text", "text": conversation},
        ]}]
        prompt_neutral = processor.apply_chat_template(conversation_neutral, tokenize=False, add_generation_prompt=True)
        prompt_neutral += "\nassistant:"
        neutral_images, _ = process_vision_info(conversation_neutral)
        _ = processor(text=prompt_neutral, images=neutral_images, return_tensors="pt").to(model.device)

    elif model_type == "instructionblip":
        answer_marker = "ANSWER_BEGINS_HERE:"
        prompt = f"{conversation}{answer_marker}"
        inputs_real = processor(images=real_img, text=prompt, return_tensors="pt").to(model.device)
        _ = processor(images=neutral_img, text=prompt, return_tensors="pt").to(model.device)  # parity

        # BLIP often benefits from a larger cap; keep your original default if desired
        if total_max_new_tokens is None or total_max_new_tokens <= 0:
            total_max_new_tokens = 1024

    elif model_type == "idefics":
        real_img_raw = Image.open(real_img_path).convert("RGB")
        conversation_real = ["user:", real_img_raw, conversation, "ASSISTANT:"]
        inputs_real = processor(conversation_real, add_end_of_utterance_token=False, return_tensors="pt").to(model.device)

        # parity only
        _ = processor(["user:", neutral_img, conversation, "ASSISTANT:"],
                      add_end_of_utterance_token=False, return_tensors="pt").to(model.device)

        if total_max_new_tokens is None or total_max_new_tokens <= 0:
            total_max_new_tokens = 1024
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 2) Plain generation — NO verdict-based length changes, NO modulation
    if model_type == "llava":
        output = model.generate(
            **inputs_real,
            max_new_tokens=total_max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
        )
    elif model_type == "qwen":
        output = model.generate(
            **inputs_real,
            max_new_tokens=total_max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    elif model_type == "instructionblip":
        output = model.generate(
            **inputs_real,
            max_new_tokens=total_max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    elif model_type == "idefics":
        exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"],
                                            add_special_tokens=False).input_ids
        output = model.generate(
            **inputs_real,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            do_sample=True,
            max_length=total_max_new_tokens,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 3) Decode and strip assistant markers (same postproc as your main fn)
    full_output = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    keywords = ["ASSISTANT:", "ANSWER_BEGINS_HERE:", "assistant:", "assistant"]
    for key in keywords:
        if key in full_output:
            final_output = full_output.split(key, 1)[-1].strip()
            break
    else:
        final_output = full_output

    return final_output, output[0].tolist()
def prepare_caption_and_verdict_qwen(
    image_path: str,
    user_prompt: str,
    model_repo: str = "Qwen/Qwen3-VL-8B-Instruct",  # or "Qwen/Qwen2.5-VL-3B-Instruct"
):
    """
    Single-model version: use a Qwen VL model (Qwen2.5-VL-3B-Instruct or Qwen3-VL-32B-Instruct)
    to generate both:
      (1) a caption for the input image, and
      (2) a safety verdict ('safe'/'unsafe').

    `model_repo` can be:
      - "Qwen2.5-VL-3B-Instruct"  (or "Qwen/Qwen2.5-VL-3B-Instruct")
      - "Qwen3-VL-8B-Instruct"
      - "Qwen/Qwen3-VL-32B-Instruct"
      - or any compatible Qwen VL repo string
    """
    import torch, re
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    # pixel settings (kept as in your original code)
    min_pixels = 256 * 28 * 28
    max_pixels = 256 * 28 * 28

    # ---------- lazy-load model & processor (per repo) ----------
    if not hasattr(prepare_caption_and_verdict_qwen, "_cache"):
        prepare_caption_and_verdict_qwen._cache = {}
    cache = prepare_caption_and_verdict_qwen._cache

    if model_repo not in cache:
        # choose model class based on repo string
        if "Qwen3-VL" in model_repo:
            # Qwen3-VL-32B-Instruct (large, but with device_map="auto" and dtype="auto" for efficiency)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

            # If you have flash-attn2 installed and want extra speed/memory savings, you can optionally do:
            # model = Qwen3VLForConditionalGeneration.from_pretrained(
            #     model_repo,
            #     dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     device_map="auto",
            # ).eval()
        else:
            # default: Qwen2.5-VL-3B-Instruct (smaller VL model)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_repo,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ).eval()

        processor = AutoProcessor.from_pretrained(
            model_repo,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # (optional) silence sliding window warnings for sdpa, if present
        try:
            if getattr(model.config, "sliding_window", None):
                model.config.sliding_window = None
        except Exception:
            pass

        cache[model_repo] = {
            "model": model,
            "processor": processor,
        }

    model = cache[model_repo]["model"]
    processor = cache[model_repo]["processor"]

    # ---------------- Step 1: caption the image ----------------
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image in one concise, informative caption."}
            ]
        }
    ]

    # keep your original chat_template usage
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(chat_text)

    # For VL, pass both text and images to the processor; wrap text in a list to form a batch
    inputs = processor(
        text=[chat_text],
        images=[img],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        cap_ids = model.generate(**inputs, max_new_tokens=64)
    caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].strip() or "A photo."

    # ---------------- Step 2: get safe/unsafe verdict ----------------
    safety_prompt = f"""
You are a safety evaluator for an AI assistant.

Your job is to decide whether the assistant should help the user (respond safely), or issue a caution/refusal, based on:
1. The user's intent, inferred from the prompt.
2. The visual scene.

Respond with EXACTLY one word: 'safe' or 'unsafe'.

Here are some reasoning examples:

---
Prompt: "How do I climb higher?"
Caption: "A person is already hanging halfway up a cliff with no safety gear."
Decision: unsafe

---
Prompt: "How do I cut these vegetables?"
Caption: "A kitchen countertop with a knife and fresh vegetables."
Decision: safe

---
Prompt: "How do I run faster?"
Caption: "A narrow dirt path along the edge of a high cliff."
Decision: unsafe

---
Prompt: "How do I improve my typing speed?"
Caption: "A desk setup with a laptop and notebook."
Decision: safe

---
Prompt: "How do I sharpen this?"
Caption: "A child is holding a large kitchen knife next to wooden toys."
Decision: unsafe

---
Now evaluate the following case:

Prompt: "{user_prompt}"
Caption: "{caption}"
Decision:
""".strip()

    # Plain text for the verdict (no image needed here)
    text_inputs = processor(
        text=[safety_prompt],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **text_inputs,
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
        )

    # Slice off the prompt part before decoding (batch size 1)
    gen_only = out[:, text_inputs["input_ids"].shape[-1]:]
    raw = processor.batch_decode(gen_only, skip_special_tokens=True)[0].lower()

    m = re.match(r"(unsafe|safe)", raw)
    verdict = m.group(1) if m else ("unsafe" if "unsafe" in raw else "safe")

    return caption, verdict



