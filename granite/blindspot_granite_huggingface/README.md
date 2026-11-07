# Blindspot — IBM Granite via Hugging Face

This starter uses **Hugging Face**, not IBM watsonx.ai. Two independent paths:

- **Hosted Granite instruction model** for structured CVE extraction. Requires an HF token and a Granite instruction model with a supported **Inference Provider**. Not every Granite model is hosted; a model page alone does not guarantee API availability.
- **Local Granite Embedding English R2** for semantic similarity. Downloads weights on first run; no token needed for the public model.

## Windows / VS Code setup

Use Python 3.10–3.12 for best library compatibility. In PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py -3.12` is unavailable, install Python 3.12 or use a compatible Python version. PyTorch may require platform-specific installation; see pytorch.org.

## A. Test embeddings first (no API key)

```powershell
python granite_embeddings.py
```

The first run downloads `ibm-granite/granite-embedding-english-r2` from Hugging Face. This may take time and disk space. Outputs are cosine similarity scores for three example descriptions.

## B. Hosted instruction-model API

1. Create a Hugging Face token at https://huggingface.co/settings/tokens with permissions required for Inference Providers.
2. Open https://huggingface.co/collections/ibm-granite/granite-42-language-models and choose an **instruction/chat** model that explicitly lists a supported Inference Provider. **If no Granite model is hosted, this API path will not work**; use local inference or another hosting service instead.
3. Copy `.env.example` to `.env` and set `HF_TOKEN` and the exact hosted `HF_MODEL` ID. Never commit `.env`.
4. Run:

```powershell
python test_granite.py
```

If you get an error about model/provider availability, the model is not hosted by your selected provider; the code cannot fix that. API requests may be billed or rate-limited.

## C. Process CSV samples

```powershell
python batch_extract.py sample_vulnerabilities.csv --limit 2
```

The input needs `cve_id` and `description`. Output is `outputs/granite_features.csv`. Start with 10 examples and manually inspect quality before running thousands. The script records per-row errors and is intended as a small starter, not a production-scale pipeline.

## Project notes

- Keep `cve_id` for joining extracted features to Person A/B's data.
- Extract only information available in the original publication-day description; don't use later NIST enrichments.
- Label 100 descriptions manually and compare extraction accuracy by field.
- Do not assume Granite's generated JSON is always correct; check missing values, schema, and hallucinations.
- For large embedding datasets, batch computation and store CVE IDs alongside vectors.
