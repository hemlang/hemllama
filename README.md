# hemllama

Hemlock bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling LLM inference in the Hemlock programming language.

## Prerequisites

- [Hemlock](https://github.com/yourusername/hemlock) programming language installed
- CMake 3.14+
- C/C++ compiler (gcc, clang)
- libffi (for Hemlock FFI)

## Building

1. Clone this repository with submodules:
```bash
git clone --recursive https://github.com/emnakamura/hemllama.git
cd hemllama
```

2. Build llama.cpp as a shared library:
```bash
./build.sh
```

For GPU support (CUDA):
```bash
./build.sh -DGGML_CUDA=ON
```

For Metal (macOS):
```bash
./build.sh -DGGML_METAL=ON
```

3. Set the library path:
```bash
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
```

## Quick Start

```hemlock
import "llama" from "path/to/hemllama/llama.hml";

// Initialize
llama.backend_init();

// Load model
let model = llama.model_load("model.gguf", {
    n_gpu_layers: 0  // Increase for GPU offloading
});

// Create context
let ctx = llama.context_create(model, {
    n_ctx: 2048
});

// Create sampler
let sampler = llama.sampler_create({
    temp: 0.8,
    top_k: 40,
    top_p: 0.9
});

// Tokenize and decode prompt
let tokens = llama.tokenize(model, "Hello, world!", { add_special: true });
llama.decode(ctx, tokens, 0);

// Generate tokens
for (let i = 0; i < 100; i = i + 1) {
    let token = llama.sample(sampler, ctx, -1);

    if (llama.is_eog(model, token)) {
        break;
    }

    let piece = llama.token_to_piece(model, token, false);
    print(piece);

    llama.decode(ctx, [token], tokens.length + i);
}

// Cleanup
llama.sampler_free(sampler);
llama.context_free(ctx);
llama.model_free(model);
llama.backend_free();
```

## High-Level API

For simple use cases, use the `complete()` function:

```hemlock
import "llama" from "hemllama/llama.hml";

let result = llama.complete("model.gguf", "What is 2+2?", {
    n_predict: 128,
    temp: 0.7
});

print(result);
```

Or use `generate()` for more control:

```hemlock
llama.backend_init();
let model = llama.model_load("model.gguf");
let ctx = llama.context_create(model);

let result = llama.generate(ctx, model, "Tell me a joke:", {
    n_predict: 256,
    temp: 0.9,
    callback: fn(token, piece) {
        print_inline(piece);
        return true;  // Continue generating
    }
});

llama.context_free(ctx);
llama.model_free(model);
llama.backend_free();
```

## API Reference

### Backend Functions

- `backend_init()` - Initialize the llama backend (call once)
- `backend_free()` - Free the backend (call once at end)
- `system_info()` - Get system info string
- `supports_gpu()` - Check if GPU offloading is supported

### Model Functions

- `model_load(path, options)` - Load a GGUF model file
  - Options: `n_gpu_layers`, `use_mmap`, `use_mlock`, `split_mode`, `main_gpu`
- `model_free(model)` - Free a loaded model
- `model_info(model)` - Get model information
- `model_vocab(model)` - Get the model's vocabulary pointer

### Context Functions

- `context_create(model, options)` - Create an inference context
  - Options: `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `embeddings`
- `context_free(ctx)` - Free a context
- `context_info(ctx)` - Get context information
- `set_threads(ctx, n, n_batch)` - Set thread counts

### Tokenization

- `tokenize(model, text, options)` - Convert text to tokens
  - Options: `add_special`, `parse_special`
- `detokenize(model, tokens, options)` - Convert tokens to text
  - Options: `remove_special`, `unparse_special`
- `token_to_piece(model, token, special)` - Convert single token to text
- `vocab_info(model)` - Get vocabulary information
- `is_eog(model, token)` - Check if token is end-of-generation

### Sampling

- `sampler_create(options)` - Create a sampler chain
  - Options: `top_k`, `top_p`, `min_p`, `temp`, `repeat_penalty`, `repeat_last_n`, `seed`
- `sampler_greedy()` - Create a greedy sampler
- `sampler_free(sampler)` - Free a sampler
- `sampler_reset(sampler)` - Reset sampler state
- `sample(sampler, ctx, idx)` - Sample a token
- `sampler_accept(sampler, token)` - Accept a token (for penalties)

### Advanced Samplers

- `sampler_init_grammar(model, grammar_str, root)` - Grammar-constrained sampling (GBNF)
- `sampler_init_logit_bias(model, biases)` - Logit bias (`[{ token, bias }]`)
- `sampler_init_infill(model)` - Fill-in-the-middle sampling
- `sampler_init_top_k(k)` - Top-k sampler
- `sampler_init_top_p(p, min_keep)` - Nucleus sampler
- `sampler_init_min_p(p, min_keep)` - Min-p sampler
- `sampler_init_temp(t)` - Temperature sampler
- `sampler_init_temp_ext(t, delta, exp)` - Dynamic temperature
- `sampler_init_typical(p, min_keep)` - Typical probability
- `sampler_init_mirostat(n_vocab, seed, tau, eta, m)` - Mirostat v1
- `sampler_init_mirostat_v2(seed, tau, eta)` - Mirostat v2
- `sampler_init_penalties(last_n, repeat, freq, presence)` - Repetition penalties
- `sampler_init_greedy()` - Greedy selection
- `sampler_init_dist(seed)` - Distribution-based selection

### Sampler Chain

- `sampler_chain_init()` - Create empty sampler chain
- `sampler_chain_add(chain, sampler)` - Add sampler to chain
- `sampler_chain_n(chain)` - Number of samplers in chain
- `sampler_chain_get(chain, i)` - Get sampler at index
- `sampler_chain_remove(chain, i)` - Remove sampler at index
- `sampler_clone(sampler)` - Clone a sampler
- `sampler_name(sampler)` - Get sampler name

### Inference

- `decode(ctx, tokens, pos)` - Decode tokens
- `decode_batch(ctx, tokens)` - Decode tokens (simpler interface)
- `get_logits(ctx)` - Get logits for last token
- `get_logits_ith(ctx, i)` - Get logits for token at index
- `synchronize(ctx)` - Wait for async operations

### KV Cache

- `kv_cache_clear(ctx)` - Clear the KV cache
- `kv_cache_seq_rm(ctx, seq_id, p0, p1)` - Remove tokens from sequence
- `kv_cache_seq_cp(ctx, src, dst, p0, p1)` - Copy a sequence
- `kv_cache_seq_keep(ctx, seq_id)` - Keep only one sequence
- `kv_cache_seq_add(ctx, seq_id, p0, p1, delta)` - Shift positions
- `kv_cache_seq_pos_min(ctx, seq_id)` - Get min position
- `kv_cache_seq_pos_max(ctx, seq_id)` - Get max position

### Embeddings

- `encode(ctx, tokens)` - Encode tokens for embedding models
- `get_embeddings(ctx)` - Get all embeddings
- `get_embeddings_ith(ctx, i)` - Get embedding at index
- `embed(ctx, model, text)` - Convenience: tokenize, encode, extract embedding
- `pooling_type(ctx)` - Get pooling type (none/mean/cls/last)

### LoRA Adapters

- `lora_adapter_init(model, path)` - Load a LoRA adapter
- `lora_adapter_free(adapter)` - Free a LoRA adapter
- `lora_adapter_set(ctx, adapter, scale)` - Apply adapter (scale: 0.0-1.0)
- `lora_adapter_remove(ctx, adapter)` - Remove a specific adapter
- `lora_adapter_clear(ctx)` - Remove all adapters

### State Save/Load

- `state_save(ctx)` - Save full context state
- `state_load(ctx, state)` - Restore full context state
- `state_free(state)` - Free saved state
- `state_seq_save(ctx, seq_id)` - Save single sequence
- `state_seq_load(ctx, state, seq_id)` - Restore single sequence
- `state_get_size(ctx)` - Get state buffer size

### Vocabulary Inspection

- `vocab_get_text(model, token)` - Get token text
- `vocab_get_score(model, token)` - Get token score
- `vocab_get_attr(model, token)` - Get token attributes
- `is_control(model, token)` - Check if control token

### Model Quantization

- `model_quantize(input, output, options)` - Quantize a model
  - Options: `ftype`, `n_threads`, `allow_requantize`, `quantize_output_tensor`

### High-Level

- `generate(ctx, model, prompt, options)` - Generate text with streaming callback
  - Options: `n_predict`, `sampler`, `stop_tokens`, `callback`
- `complete(model_path, prompt, options)` - One-shot completion

### Performance

- `perf_print(ctx)` - Print context performance stats
- `perf_reset(ctx)` - Reset performance stats
- `perf_sampler_print(sampler)` - Print sampler performance stats
- `perf_sampler_reset(sampler)` - Reset sampler performance stats
- `time_us()` - Get current time in microseconds

## Examples

See the `examples/` directory:

| Example | Description |
|---------|-------------|
| `simple.hml` | Basic text generation |
| `chat.hml` | Interactive chat interface |
| `oneshot.hml` | One-shot completion using `complete()` |
| `streaming.hml` | Streaming generation with performance metrics |
| `chat_template.hml` | Chat template formatting (ChatML, Llama, etc.) |
| `grammar.hml` | Grammar-constrained JSON generation |
| `embeddings.hml` | Text embeddings and similarity comparison |
| `tokenizer.hml` | Tokenizer/vocabulary exploration |
| `infill.hml` | Code infill / fill-in-the-middle (FIM) |
| `model_info.hml` | Model metadata and system inspection |

Run examples:
```bash
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"

# Basic generation
hemlock examples/simple.hml ~/models/llama-7b.gguf "What is the meaning of life?"

# Interactive chat
hemlock examples/chat.hml ~/models/llama-7b-chat.gguf

# Grammar-constrained JSON output
hemlock examples/grammar.hml ~/models/llama-7b.gguf "Describe a fictional character"

# Text embeddings
hemlock examples/embeddings.hml ~/models/nomic-embed-text.gguf

# Model inspection
hemlock examples/model_info.hml ~/models/llama-7b.gguf
```

## License

MIT License - see llama.cpp for its license terms.
