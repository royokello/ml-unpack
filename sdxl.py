import argparse
import os
import sys
import torch
from safetensors.torch import load_file, save_file
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_precision(state_dict, precision):
    """
    Convert the precision of tensors in the state dictionary.

    Args:
        state_dict (dict): The original state dictionary with tensor values.
        precision (str): The target precision format ('fp16', 'fp32', 'int8', 'int4').

    Returns:
        dict: The state dictionary with tensors converted to the specified precision.
    """
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        try:
            if precision == "fp16":
                converted_state_dict[key] = tensor.to(torch.float16)
            elif precision == "fp32":
                converted_state_dict[key] = tensor.to(torch.float32)
            elif precision == "int8":
                converted_state_dict[key] = tensor.to(torch.int8)
            elif precision == "int4":
                # Simulate INT4 by using INT8 (since PyTorch does not natively support INT4)
                # Note: This is a simplistic simulation and may not be suitable for all use cases
                converted_state_dict[key] = (tensor.to(torch.int8) // 16)
            else:
                raise ValueError(f"Unsupported precision format: {precision}")
        except Exception as e:
            logger.error(f"Error converting tensor {key} to {precision}: {e}")
            raise
    return converted_state_dict

def sdxl(input_path: str, output_path: str, precision: str = None):
    """
    Convert the precision of model components and save them separately.

    Args:
        input_path (str): Path to the input safetensors model file.
        output_path (str): Directory where the converted model components will be saved.
        precision (str, optional): Target precision format. If None, no conversion is performed.
    """
    # Load the safetensors model
    try:
        state_dict = load_file(input_path)
        logger.info(f"Loaded model from {input_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {input_path}: {e}")
        sys.exit(1)

    # Initialize dictionaries to store components
    unet_state_dict = {}
    vae_state_dict = {}
    conditioner_state_dict = {}

    # Split the model components
    for key, value in state_dict.items():
        if key.startswith('model.diffusion_model'):
            unet_state_dict[key] = value
        elif key.startswith('first_stage_model'):
            vae_state_dict[key] = value
        elif key.startswith('conditioner.embedders'):
            conditioner_state_dict[key] = value

    # Convert to the specified precision format if precision is provided
    if precision:
        try:
            unet_state_dict = convert_precision(unet_state_dict, precision)
            vae_state_dict = convert_precision(vae_state_dict, precision)
            conditioner_state_dict = convert_precision(conditioner_state_dict, precision)
            logger.info(f"Converted model components to {precision}")
        except Exception as e:
            logger.error(f"Precision conversion failed: {e}")
            sys.exit(1)
    else:
        logger.info("No precision conversion specified. Retaining original tensor precisions.")

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory is set to {output_path}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_path}: {e}")
        sys.exit(1)

    # Define a helper function to save components
    def save_component(component_dict, name):
        filename = f'{name}'
        if precision:
            filename += f'_{precision}'
        filename += '.safetensors'
        file_path = os.path.join(output_path, filename)
        try:
            save_file(component_dict, file_path)
            logger.info(f"Saved {name} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {name} to {file_path}: {e}")
            sys.exit(1)

    # Save each component
    save_component(unet_state_dict, 'unet')
    save_component(vae_state_dict, 'vae')
    save_component(conditioner_state_dict, 'conditioner')

    logger.info(f"All components saved successfully in {output_path}")

if __name__ == "__main__":
    """
    Usage: python sdxl.py -i input_model_path -o output_directory_path [-p precision_format]
    """
    parser = argparse.ArgumentParser(description="Convert model precision and segregate components.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input safetensors model file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory to save the converted model components.")
    parser.add_argument("-p", "--precision", type=str, choices=["fp16", "fp32", "int8", "int4"],
                        help="Precision format to convert tensors to (choices: fp16, fp32, int8, int4). If not specified, tensors retain their original precision.")
    args = parser.parse_args()

    sdxl(args.input, args.output, args.precision)
