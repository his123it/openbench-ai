# Local Ollama Integration Guide (v0.3.0)

`OllamaAdapter` (`openbench.adapters.OllamaAdapter`) connects OpenBench AI to a local [Ollama](https://ollama.com/) server without requiring any paid API keys or third-party Python HTTP libraries.

---

## 1. Prerequisites

1. Install Ollama locally and start the server:
   ```bash
   ollama serve
   ```
2. Pull the model you want to evaluate:
   ```bash
   ollama pull llama3.2
   ```
3. Verify connectivity from OpenBench AI:
   ```bash
   openbench providers check ollama --model llama3.2
   ```

*(Note: OpenBench AI never assumes Ollama is installed or running. If the server is unreachable, `openbench` reports a clear `OllamaConnectionError` with remediation steps.)*

---

## 2. Running an Ollama Experiment

### Via YAML Configuration Profile (`configs/local_ollama.yaml`)
```bash
openbench run --config configs/local_ollama.yaml
```

### Via CLI Flags
```bash
openbench run   --dataset datasets/reasoning/sample.jsonl   --provider ollama   --model llama3.2   --endpoint http://localhost:11434   --timeout 60.0   --concurrency 2
```

---

## 3. Supported Features

- **Generation Options**: Maps `temperature`, `max_tokens` (`num_predict`), `top_p`, `seed`, `stop_sequences` (`stop`), and `extra_options` to Ollama's `/api/generate` payload.
- **Provider-Reported Token Accounting**: Automatically extracts `prompt_eval_count` (`prompt_tokens`) and `eval_count` (`completion_tokens`) from Ollama responses.
- **Streaming Support**: Set `stream: true` in `GenerationConfig` to consume NDJSON chunks incrementally.
- **Endpoint Security Validation**: Rejects unsafe URL schemes (`file://`) and forbids plain `http://` to non-loopback remote hosts unless `allow_remote_http=True` is explicitly set.
