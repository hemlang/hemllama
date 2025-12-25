# hemllama

Hemlock bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling LLM inference in the Hemlock programming language.

## Features

- **Full llama.cpp API** - Access to model loading, tokenization, inference, and sampling
- **Grammar Constraints** - Generate structured output (JSON, etc.) with GBNF grammars
- **LoRA Adapters** - Load and apply LoRA fine-tuned adapters
- **Session Management** - Save and restore conversation state
- **Embeddings** - Extract text embeddings for semantic similarity
- **FIM Support** - Fill-in-Middle code completion
- **Chat Templates** - Format conversations using model chat templates
- **GPU Acceleration** - CUDA and Metal support

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

## Grammar-Constrained Generation

Generate valid JSON or other structured output:

```hemlock
// Use built-in JSON grammar
let sampler = llama.sampler_create_with_grammar(model, {
    grammar: llama.GRAMMAR_JSON,
    grammar_root: "root",
    temp: 0.7
});

// Or define custom grammar
let custom_grammar = "root ::= \"yes\" | \"no\"";
let sampler = llama.sampler_create_with_grammar(model, {
    grammar: custom_grammar,
    temp: 0.1
});
```

## LoRA Adapters

Apply fine-tuned LoRA adapters:

```hemlock
// Load adapter
let lora = llama.lora_load(model, "adapter.gguf");

// Apply with full strength
llama.lora_apply(ctx, lora, 1.0);

// Or partial strength
llama.lora_apply(ctx, lora, 0.5);

// Remove adapter
llama.lora_clear(ctx);
llama.lora_free(lora);
```

## Session Management

Save and restore conversation state:

```hemlock
// Save session
let tokens = [...]; // collected tokens
llama.state_save(ctx, "session.bin", tokens);

// Load session later
let loaded = llama.state_load(ctx, "session.bin", 4096);
if (loaded.success) {
    print("Loaded " + loaded.tokens.length + " tokens");
}
```

## Chat Templates

Format conversations using model templates:

```hemlock
let messages = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" }
];

let formatted = llama.chat_format(model, messages, true);
print(formatted);  // Formatted prompt with model's template
```

## Fill-in-Middle (Code Completion)

```hemlock
if (llama.supports_fim(model)) {
    let prompt = llama.fim_prompt(model,
        "def add(a, b):\n    ",     // prefix
        "\n    return result"        // suffix
    );
    // Generate completion for the middle
}
```

## Embeddings

Extract text embeddings:

```hemlock
let ctx = llama.context_create(model, {
    embeddings: true,
    pooling_type: llama.POOLING_TYPE_MEAN
});

let tokens = llama.tokenize(model, "Hello world", { add_special: true });
llama.decode(ctx, tokens, 0);

let embeddings = llama.extract_embeddings(ctx, model, 0);
print("Embedding dimension: " + embeddings.length);
```

## API Reference

### Backend Functions

- `backend_init()` - Initialize the llama backend (call once)
- `backend_free()` - Free the backend (call once at end)
- `system_info()` - Get system info string
- `supports_gpu()` - Check if GPU offloading is supported
- `supports_rpc()` - Check if RPC is supported
- `numa_init(numa)` - Initialize NUMA optimizations

### Model Functions

- `model_load(path, options)` - Load a GGUF model file
  - Options: `n_gpu_layers`, `use_mmap`, `use_mlock`, `split_mode`, `main_gpu`
- `model_free(model)` - Free a loaded model
- `model_info(model)` - Get model information
- `model_info_extended(model)` - Get extended model info (hybrid, diffusion, RoPE, etc.)
- `model_vocab(model)` - Get the model's vocabulary pointer
- `model_meta(model, key)` - Get metadata value by key
- `model_meta_all(model)` - Get all metadata as object
- `chat_template(model)` - Get model's chat template

### Context Functions

- `context_create(model, options)` - Create an inference context
  - Options: `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `embeddings`, `n_seq_max`
- `context_free(ctx)` - Free a context
- `context_info(ctx)` - Get context information
- `set_threads(ctx, n, n_batch)` - Set thread counts
- `set_embeddings(ctx, enabled)` - Enable/disable embeddings
- `set_causal_attn(ctx, causal)` - Set causal attention mode
- `set_warmup(ctx, warmup)` - Set warmup mode
- `n_ctx_seq(ctx)` - Get sequence context size

### Tokenization

- `tokenize(model, text, options)` - Convert text to tokens
  - Options: `add_special`, `parse_special`
- `detokenize(model, tokens, options)` - Convert tokens to text
  - Options: `remove_special`, `unparse_special`
- `token_to_piece(model, token, special)` - Convert single token to text
- `vocab_info(model)` - Get vocabulary information
- `is_eog(model, token)` - Check if token is end-of-generation
- `token_score(model, token)` - Get token score
- `token_attr(model, token)` - Get token attributes
- `vocab_sep(model)` - Get separator token
- `vocab_mask(model)` - Get mask token

### FIM (Fill-in-Middle)

- `supports_fim(model)` - Check if model supports FIM
- `fim_tokens(model)` - Get FIM special tokens
- `fim_prompt(model, prefix, suffix)` - Create FIM prompt

### Sampling

- `sampler_create(options)` - Create a sampler chain
  - Options: `top_k`, `top_p`, `min_p`, `temp`, `repeat_penalty`, `repeat_last_n`, `seed`
- `sampler_create_with_grammar(model, options)` - Create sampler with grammar
  - Additional options: `grammar`, `grammar_root`
- `sampler_grammar(model, grammar_str, grammar_root)` - Create grammar-only sampler
- `sampler_greedy()` - Create a greedy sampler
- `sampler_xtc(p, t, min_keep, seed)` - Create XTC sampler
- `sampler_top_n_sigma(n)` - Create Top-N-Sigma sampler
- `sampler_infill(model)` - Create infill sampler for FIM
- `sampler_free(sampler)` - Free a sampler
- `sampler_reset(sampler)` - Reset sampler state
- `sample(sampler, ctx, idx)` - Sample a token
- `sampler_accept(sampler, token)` - Accept a token (for penalties)
- `GRAMMAR_JSON` - Built-in JSON grammar string

### LoRA Adapters

- `lora_load(model, path)` - Load a LoRA adapter
- `lora_free(adapter)` - Free a LoRA adapter
- `lora_apply(ctx, adapter, scale)` - Apply adapter to context
- `lora_remove(ctx, adapter)` - Remove specific adapter
- `lora_clear(ctx)` - Remove all adapters

### State/Sessions

- `state_size(ctx)` - Get state size in bytes
- `state_save(ctx, path, tokens)` - Save state to file
- `state_load(ctx, path, max_tokens)` - Load state from file
- `state_seq_save(ctx, path, seq_id, tokens)` - Save single sequence
- `state_seq_load(ctx, path, seq_id, max_tokens)` - Load single sequence

### Embeddings

- `get_embeddings(ctx, seq_id)` - Get embeddings pointer
- `extract_embeddings(ctx, model, seq_id)` - Extract embeddings as array
- `get_pooling_type(ctx)` - Get pooling type

### Chat Templates

- `chat_template(model)` - Get model's chat template string
- `chat_format(model, messages, add_assistant)` - Format messages array

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
- `kv_cache_seq_div(ctx, seq_id, p0, p1, d)` - Divide positions
- `kv_cache_seq_pos_min(ctx, seq_id)` - Get min position
- `kv_cache_seq_pos_max(ctx, seq_id)` - Get max position
- `memory_can_shift(ctx)` - Check if memory supports shifting

### High-Level

- `generate(ctx, model, prompt, options)` - Generate text
  - Options: `n_predict`, `sampler`, `stop_tokens`, `callback`
- `complete(model_path, prompt, options)` - One-shot completion

### Performance

- `perf_print(ctx)` - Print context performance stats
- `perf_reset(ctx)` - Reset performance stats
- `perf_sampler_print(sampler)` - Print sampler stats
- `time_us()` - Get current time in microseconds

### Constants

- Token types: `TOKEN_TYPE_NORMAL`, `TOKEN_TYPE_CONTROL`, etc.
- Token attributes: `TOKEN_ATTR_NORMAL`, `TOKEN_ATTR_CONTROL`, etc.
- Vocab types: `VOCAB_TYPE_SPM`, `VOCAB_TYPE_BPE`, `VOCAB_TYPE_WPM`, etc.
- Split modes: `SPLIT_MODE_NONE`, `SPLIT_MODE_LAYER`, `SPLIT_MODE_ROW`
- Pooling types: `POOLING_TYPE_NONE`, `POOLING_TYPE_MEAN`, `POOLING_TYPE_CLS`, `POOLING_TYPE_LAST`
- RoPE scaling: `ROPE_SCALING_NONE`, `ROPE_SCALING_LINEAR`, `ROPE_SCALING_YARN`, `ROPE_SCALING_LONGROPE`

## Examples

See the `examples/` directory:

- `simple.hml` - Basic text generation
- `chat.hml` - Interactive chat interface
- `oneshot.hml` - One-shot completion
- `json_output.hml` - Grammar-constrained JSON generation
- `lora.hml` - LoRA adapter usage
- `embeddings.hml` - Text embeddings extraction
- `fim.hml` - Fill-in-Middle code completion
- `session.hml` - Save/restore conversation sessions

Run examples:
```bash
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
hemlock examples/simple.hml ~/models/llama-7b.gguf "What is the meaning of life?"
hemlock examples/json_output.hml ~/models/llama.gguf "List 3 colors"
hemlock examples/chat.hml ~/models/llama.gguf
```

## License

MIT License - see llama.cpp for its license terms.
