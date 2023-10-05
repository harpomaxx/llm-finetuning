import argparse
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from transformers import pipeline, set_seed
from transformers import TextStreamer, pipeline
from pynvml import *
    

def generate_text(prompt, generator, template):
    formatted_prompt = template.format(prompt=prompt)
    _ = generator(formatted_prompt)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate text using a given model.")
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m", help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, default="facebook/opt-350m", help="Path to the pretrained tokenizer.")
    parser.add_argument("--prompt", type=str, default="""Code a function in PYTHON language for calculating prime numbers ### Answer:""", help="Prompt for the generator.")
    parser.add_argument("--cli", action="store_true", help="Activate interactive CLI mode.")
    parser.add_argument("--quantize", action="store_true", default = False, help="Load the model using quantization.")
    parser.add_argument("--template", type=str, default="### Human {prompt} ### Assistant", help="Template for the prompt.")

    args = parser.parse_args()

    print(f"[] Selected Model Path: {args.model_path}")
    print(f"[] Selected Tokenizer Path: {args.tokenizer_path}")
    print(f"[] Using CLI Mode: {'Yes' if args.cli else 'No'}")
    print(f"[] Using Quantization: {'Yes' if args.quantize else 'No'}")
    print_gpu_utilization()
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype= getattr(torch, "float16"),
    bnb_4bit_use_double_quant=True,
    )
   
    quantization_config = bnb_config if args.quantize else None


    model_code = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map ="auto",
            #device_map={"": 0},
            trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    set_seed(32)
    print_gpu_utilization()
    print(f"Selected prompt: {args.prompt}")
    print(f"Answer:\n")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    generator = pipeline('text-generation', 
                         model=model_code, 
                         tokenizer=tokenizer, 
                         streamer=streamer, 
                         do_sample=True,
                         max_length=256)

    if args.cli:
        while True:
            prompt = input("Enter your prompt (or type 'exit' to quit): ")
            if prompt.lower() == 'exit':
                break
            generate_text(prompt, generator, args.template)
    else:
        generate_text(args.prompt, generator, args.template)
