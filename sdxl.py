import argparse
import os
from safetensors.torch import load_file, save_file

def sdxl(input_path: str, output_path: str):
    # Load the safetensors model
    print(f"Loading model from {input_path}...")
    state_dict = load_file(input_path)

    # Initialize dictionaries to store components
    print("Splitting model components...")
    unet_state_dict = {}
    vae_state_dict = {}
    conditioner_state_dict = {}

    # Split the model components
    for key, value in state_dict.items():
        # UNet components
        if key.startswith('model.diffusion_model'):
            unet_state_dict[key] = value
        # VAE components (encoder and decoder, including quantization)
        elif key.startswith('first_stage_model'):
            vae_state_dict[key] = value
        # Conditioner components (embedders, possibly CLIP or text embeddings)
        elif key.startswith('conditioner.embedders'):
            conditioner_state_dict[key] = value

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save each component into separate safetensors files
    print("Saving components...")
    save_file(unet_state_dict, os.path.join(output_path, 'unet.safetensors'))
    save_file(vae_state_dict, os.path.join(output_path, 'vae.safetensors'))
    save_file(conditioner_state_dict, os.path.join(output_path, 'conditioner.safetensors'))

    print(f"Components saved to {output_path} as unet.safetensors, vae.safetensors, and conditioner.safetensors")

if __name__ == "__main__":
    """
    usage: python sdxl.py -i input_model_path -o output_directory_path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input model path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    sdxl(args.input, args.output)
