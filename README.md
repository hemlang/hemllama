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

### High-Level

- `generate(ctx, model, prompt, options)` - Generate text
  - Options: `n_predict`, `sampler`, `stop_tokens`, `callback`
- `complete(model_path, prompt, options)` - One-shot completion

### Performance

- `perf_print(ctx)` - Print context performance stats
- `perf_reset(ctx)` - Reset performance stats
- `time_us()` - Get current time in microseconds

## Examples

See the `examples/` directory:

- `simple.hml` - Basic text generation
- `chat.hml` - Interactive chat interface
- `oneshot.hml` - One-shot completion

Run examples:
```bash
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
hemlock examples/simple.hml ~/models/llama-7b.gguf "What is the meaning of life?"
```

## License

MIT License - see llama.cpp for its license terms.
