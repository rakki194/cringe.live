def calculate_memory_per_token(hidden_size, hidden_layers, key_value_heads, attention_heads):
    # Constants
    BYTES_PER_TOKEN = 2
    
    # Calculate memory per token in bytes
    memory = (2 * BYTES_PER_TOKEN) * (2) * hidden_layers * key_value_heads * hidden_size / attention_heads
    
    # Convert to kilobytes
    memory_kb = memory / 1024
    
    return memory_kb

def calculate_total_memory(context_length, memory_per_token):
    # Calculate total memory for the given context length
    total_memory_kb = memory_per_token * context_length
    total_memory_mb = total_memory_kb / 1024
    total_memory_gb = total_memory_mb / 1024
    
    return total_memory_kb, total_memory_mb, total_memory_gb

def main():
    # Model architecture parameters
    params = {
        'hidden_size': 4096,
        'hidden_layers': 32,
        'key_value_heads': 8,
        'attention_heads': 32
    }
    
    # Calculate memory per token
    memory_per_token = calculate_memory_per_token(**params)
    print(f"Memory per token: {memory_per_token:.2f} KB")
    
    # Context lengths
    context_lengths = [4096, 8192, 16384, 32768, 65536]
    
    # Calculate and display total memory requirements for different context lengths
    print("\nTotal memory requirements for different context lengths:")
    for length in context_lengths:
        memory_kb, memory_mb, memory_gb = calculate_total_memory(length, memory_per_token)
        print(f"\nContext Length: {length}")
        print(f"Memory required: {memory_kb:.2f} KB")
        print(f"               = {memory_mb:.2f} MB")
        print(f"               = {memory_gb:.2f} GB")

if __name__ == "__main__":
    main()
