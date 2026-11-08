"""
Runs IBM Granite 4.0 Micro (a small chat model) on machine with GPU.

explain.py uses chat() to turn facts into plain English.
extract() pulls vendor / product / login needed / impact out of a description
(handy for an accuracy check, not used by the main pipeline).

On a GPU it loads squished to 4-bit so it fits in ~3 GB, ~7 s per answer.
No GPU? Still works, just slowly. The laptops never need this, since results are saved to files.

Run:  python src/granite/local_granite.py      (quick test)
"""

import json
import re

MODEL_ID = "ibm-granite/granite-4.0-micro"

EXTRACT_PROMPT = """Extract facts explicitly stated in this vulnerability description.
Return ONLY one JSON object with keys:
vendor, product, product_type (one of: vpn, firewall, edge_device, mail_server, web_app,
cms_plugin, os, browser, library, database, network_device, mobile_app, other),
auth_required (true, false or null), impact (one of: code_execution, auth_bypass,
privilege_escalation, info_leak, denial_of_service, other), referenced_cves (list).
Use null when not stated. Do not use outside knowledge."""

_tok = _model = None  # loaded once, on first use

def _load():
    global _tok, _model
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        four_bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type="nf4")
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=four_bit,
                                                      device_map="cuda:0")
    else:
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    _model.eval()


def chat(messages, max_tokens=300):
    """Send a conversation, get Granite's reply back as text."""
    import torch
    _load()
    ids = _tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True).to(_model.device)
    with torch.no_grad():
        out = _model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract(description):
    """Description in, dict of facts out."""
    out = chat([{"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": description}])
    m = re.search(r"\{.*\}", out, re.S)  # grab the JSON even if Granite adds extra words
    if not m:
        raise ValueError(f"no JSON in reply: {out[:200]}")
    return json.loads(m.group(0))


if __name__ == "__main__":
    import time
    t = time.time()
    _load()
    print(f"loaded in {time.time()-t:.0f}s")
    sample = ("A stack-based buffer overflow in Fortinet FortiOS SSL-VPN allows a remote "
              "unauthenticated attacker to execute arbitrary code via crafted HTTP requests.")
    t = time.time()
    print(json.dumps(extract(sample), indent=2))
    print(f"{time.time()-t:.1f}s per extraction")
