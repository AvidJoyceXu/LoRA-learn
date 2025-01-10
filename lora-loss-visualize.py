import os
import json

def compare_loss(results_dir, key="train"):
    '''
    Read the lora-$dim directory, with dim = 1,2,4,8,16,32,
    Read the train_data.json file which is formatted as:
        {
        "train_steps": [...],
        "train_loss":  [...],
        "eval_loss":  [...],
        }
    visualize and compare the train and eval loss with different lora dimensions.
    '''
    import matplotlib.pyplot as plt

    dims = [1, 2, 4, 8, 16, 32]
    steps = None
    losses = {}

    for dim in dims:
        dir_path = os.path.join(results_dir, f'lora-{dim}')
        file_path = os.path.join(dir_path, 'train_data.json')
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            if steps is None:
                steps = data[f'{key}_steps']
            losses[dim] = data[f'{key}_loss']

    plt.figure(figsize=(12, 6))
    
    for dim in dims:
        if dim in losses:
            plt.plot(steps, losses[dim], label=f'{key} Loss (dim={dim})')
            # plt.plot(train_steps, eval_losses[dim], label=f'Eval Loss (dim={dim})', linestyle='--')
    
    plt.xlabel(f'{key} steps')
    plt.ylabel('Loss')
    plt.title(f'{key} loss for different LoRA dimensions')
    plt.legend()
    plt.grid(True)
    plt.show()
    output_path = os.path.join(results_dir, f'{key}_loss_comparison.png')
    plt.savefig(output_path)
    print(f"Loss comparison plot saved to {output_path}")

compare_loss("results", "train")
compare_loss("results", "eval")

