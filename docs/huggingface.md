# Hugging Face Integration Guide (v0.3.0 — Experimental)

`HuggingFaceAdapter` (`openbench.adapters.HuggingFaceAdapter`) provides an optional integration for running local Hugging Face `transformers` text-generation models.

---

## 1. Installation & Optional Dependencies

Core OpenBench AI does not require `transformers` or `torch`. Install them only when needed:

```bash
python -m pip install -e ".[huggingface]"
```

---

## 2. Offline-First & Safety Guarantees

- **No Automatic Large Model Downloads**: `HuggingFaceAdapter` sets `local_files_only=True` by default (`allow_download=False`). It will only load model weights that you have already downloaded or cached locally unless you explicitly opt in with `allow_download: true`.
- **No Hardcoded Tokens**: Zero API tokens are embedded in the codebase.
- **Explicit Revision Tracking**: Records `model_id`, `revision` (branch, tag, or commit SHA), and `device` (`"cpu"`, `"cuda"`, `"mps"`) in `model_metadata.json` and `reproducibility.json`.

---

## 3. Running a Local Hugging Face Experiment

```bash
openbench run --config configs/huggingface_local.yaml
```

---

## 4. Hardware & Compatibility Limitations

- Running multi-billion-parameter weights on CPU can be slow or exceed system RAM. `HuggingFaceAdapter` catches `MemoryError` and surfaces a clear diagnostic message.
- Marked `is_experimental = True` in `v0.3.0` because local hardware and weight availability vary across environments.
