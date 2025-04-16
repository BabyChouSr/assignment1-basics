from cs336_basics.layers.model import Model

GPT2_SMALL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 12,
    "d_model": 768,
    "num_heads": 12,
    "d_ff": 3072,
}

GPT2_MEDIUM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 36,
    "d_model": 1024,
    "num_heads": 16,
    "d_ff": 4096,
}

GPT2_LARGE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 36,
    "d_model": 1280,
    "num_heads": 20,
    "d_ff": 5120,
}

GPT2_XL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "num_layers": 48,
    "d_model": 1600,
    "num_heads": 25,
    "d_ff": 6400,
}

def num_trainable_parameters(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    model = Model(d_model, num_heads, d_ff, 10000, vocab_size, context_length, num_layers)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def num_forward_pass_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # q, k, v
    qkv_flops = 3 * (2 * context_length * d_model * d_model)
    # (qk/sqrt(d)) * v
    key_dimension = d_model // num_heads
    attn_matmul_flops = 2 * d_model * (context_length**2) + 2 * num_heads * context_length**2 * key_dimension
    # o(v)
    project_flops = 2 * context_length * d_model ** 2


    # FFN
    # w1(x)
    w1_flops = 2 * context_length * d_model * d_ff

    #  w3(x)
    w3_flops = 2 * context_length * d_model * d_ff

    # w2(x)
    w2_flops = 2 * context_length * d_model * d_ff

    total_attn_flops_per_layer = qkv_flops + attn_matmul_flops + project_flops
    total_ffn_flops_per_layer = w1_flops + w2_flops + w3_flops
    total_flops_per_layer = total_attn_flops_per_layer + total_ffn_flops_per_layer
    total_flops = total_flops_per_layer * num_layers

    return {
        "total_attn_flops_per_layer": total_attn_flops_per_layer,
        "total_ffn_flops_per_layer": total_ffn_flops_per_layer,
        "total_flops_per_layer": total_flops_per_layer,
        "total_flops": total_flops,
        "total_attn_flops_proportion": total_attn_flops_per_layer / total_flops_per_layer,
        "total_ffn_flops_proportion": total_ffn_flops_per_layer / total_flops_per_layer,
    }

def peak_memory_usage(batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # Define variables for readability
    L = num_layers
    D = d_model
    H = num_heads
    S = context_length
    V = vocab_size
    B = batch_size
    
    # Model parameters (16-bit storage)
    model_params_memory = 16 * (L * (4 * D**2 + 12 * D**2 + 2 * D) + 2 * V * D + D)
    
    # Activations and temporary buffers (32-bit storage)
    activations_memory = 4 * (L * (24 * B * S * D + 2 * B * H * S**2) + B * S * D + B * S * V)
    
    print(model_params_memory)
    print(activations_memory // batch_size)

    # Total peak memory usage in bytes
    total_memory = model_params_memory + activations_memory
    
    return total_memory
    

if __name__ == "__main__":
    # trainable_parameters = num_trainable_parameters(**GPT2_XL_CONFIG)
    # num_bytes_for_model = 4 * trainable_parameters
    # print(f"Number of trainable parameters: {trainable_parameters}")
    # print(f"Number of bytes for model: {num_bytes_for_model}")


    flops_dict = num_forward_pass_flops(**GPT2_SMALL_CONFIG)
    print(f"=" * 10 + "FLOPS FOR GPT2-SMALL" + "=" * 10)
    print(flops_dict)
    print(f"=" * 50)

    flops_dict = num_forward_pass_flops(**GPT2_MEDIUM_CONFIG)
    print(f"=" * 10 + "FLOPS FOR GPT2-MEDIUM" + "=" * 10)
    print(flops_dict)
    print(f"=" * 50)

    flops_dict = num_forward_pass_flops(**GPT2_LARGE_CONFIG)
    print(f"=" * 10 + "FLOPS FOR GPT2-LARGE" + "=" * 10)
    print(flops_dict)
    print(f"=" * 50)

    flops_dict = num_forward_pass_flops(**GPT2_XL_CONFIG)
    print(f"=" * 10 + "FLOPS FOR GPT2-XL" + "=" * 10)
    print(flops_dict)
    print(f"=" * 50)

    gpt2_xl_config_extended_context = GPT2_XL_CONFIG.copy()
    gpt2_xl_config_extended_context["context_length"] = 16384
    flops_dict = num_forward_pass_flops(**gpt2_xl_config_extended_context)
    print(f"=" * 10 + "FLOPS FOR GPT2-XL-Extended-Context" + "=" * 10)
    print(flops_dict)
    print(f"=" * 50)

    print(peak_memory_usage(batch_size=2, **GPT2_XL_CONFIG))




