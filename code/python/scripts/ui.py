import gradio as gr
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,LlamaTokenizer,LlamaForCausalLM)
from transformers import TextIteratorStreamer, set_seed
from threading import Thread
import torch
from pynvml import *

def generate(text = "", 
             max_new_tok = 128, 
             temp = 0.1,
             repet_penalty=1.0,
             do_samp = True,
             t_k = 40,
             t_p = 0.97,
             early_stop = True,
             n_beams = 1,
             no_rep_ngram_s = 4
             ):  
  streamer = TextIteratorStreamer(tok, skip_prompt=True, timeout=40.)
  if len(text) == 0:
    text = " "
  inputs = tok([text], return_tensors="pt").to("cuda")
  generation_kwargs = dict(inputs, streamer=streamer, 
                           repetition_penalty=repet_penalty *1.0,
                           do_sample= do_samp,
                           top_k= t_k,
                           top_p= t_p,
                           max_new_tokens=max_new_tok, 
                           #pad_token_id = model.config.eos_token_id, 
                           num_beams = n_beams,
                           early_stopping= early_stop,
                           no_repeat_ngram_size= no_rep_ngram_s,
                           temperature = temp )
  thread = Thread(target=model_code.generate, kwargs=generation_kwargs)
  thread.start()
  generated_text = ""
  for new_text in streamer:
    yield generated_text + new_text    
    generated_text += new_text
    if tok.eos_token in generated_text:
      generated_text = generated_text[: generated_text.find(tok.eos_token) if tok.eos_token else None]
      streamer.end()
      yield generated_text
      return
  return generated_text


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
    parser.add_argument("--quantize", action="store_true", default = False, help="Load the model using quantization.")
   
    args = parser.parse_args()

    print(f"[] Selected Model Path: {args.model_path}")
    print(f"[] Selected Tokenizer Path: {args.tokenizer_path}")
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
            #device_map ="",
            device_map={"": 0},
            trust_remote_code=True
    )
 
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True,  device_map={"": 0})

    # unwind broken decapoda-research config
    model_code.config.pad_token_id = tok.pad_token_id = 0  # unk
    model_code.config.bos_token_id = 1
    model_code.config.eos_token_id = 2

    set_seed(32)
    print_gpu_utilization()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpu = 0 if torch.cuda.is_available()==False else torch.cuda.device_count()
    #model_code.to(device)
    #tok.to(device)


    # Gradio UI components and layout
    with gr.Blocks() as demo:
        #examples = [
        #    ["Once upon a time in a world where AI", 2, 10],
        #    ["The future of AI lies in", 1, 15],
        #    ["Deep in the heart of the forest", 3, 5]
        #]

        gr.Markdown(
        f"""
        # Text Generation Demo

        ## Enter your prompt and generate text based on your input.

        ### Model:{args.model_path}. Using Quantization: {'Yes' if args.quantize else 'No'}
        """
        )
        with gr.Column(variant="panel"):
            with gr.Row(variant="compact"):
                input_text = gr.Textbox(
                    label="Enter your prompt here",
                    show_label=True,
                    lines=20,
                    #container=False
                )
                
                output_text = gr.Textbox(
                    readonly=False, 
                    label="Generated Text", 
                    lines=20,
                    #live=True,
                    #interactive=False,
                    #container=False
                )
                
            btn = gr.Button("Generate Text",full_width = True)
            
            with gr.Row(variant="compact"):
                with gr.Column():
                    gr.Markdown("**Determines if sampling should be used in generation.**")
                    do_sample_checkbox = gr.Checkbox(
                        label="Do Sample",
                        value="True"
                    )
                with gr.Column():
                    gr.Markdown("**Stops generation early if end token is found or max length reached. Set num_beans >1. (NOT SUPPPORTED)**")
                    early_stopping_checkbox = gr.Checkbox(
                        label="Early Stopping"
                    )


                with gr.Column():
                    gr.Markdown("**The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens.**")
                    max_length_slider = gr.Slider(
                        minimum=1,
                        maximum=1024,
                        step=1,
                        value=512,
                        label="Max Length",
                    )
            with gr.Row(variant="compact"):
                with gr.Column():
                    gr.Markdown("**Penalty applied when the model predicts tokens it has already predicted.**")
                    repetition_penalty_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Repetition Penalty",
                    )

               

                with gr.Column():
                    gr.Markdown("**Top K tokens considered for filtering when generating.**")
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=40,
                        label="Top K"
                    )
                with gr.Column():
                    gr.Markdown("**Number of beams for beam search. 1 means no beam search. (NOT SUPPPORTED)**")
                    n_beams_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                        label="num_beams"
                    )    

            with gr.Row(variant="compact"):
                with gr.Column():
                    gr.Markdown("**Cumulative probability for top-p-filtering in generation.**")
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.97,
                        label="Top P"
                    )

               

                with gr.Column():
                    gr.Markdown("**Ensures no n-gram of this size is repeated in the generated text.**")
                    no_repeat_ngram_size_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=4,
                        label="No Repeat N-Gram Size"
                    )

                with gr.Column():
                    gr.Markdown("**Controls randomness. Lower values make output more deterministic.**")
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=0.1,
                        label="Temperature",
                    )

        gr.on(
        triggers = [input_text.submit,btn.click],
        fn = generate,
        inputs= [input_text, 
            max_length_slider, 
            temperature_slider, 
            repetition_penalty_slider, 
            do_sample_checkbox, 
            top_k_slider, 
            top_p_slider, 
            early_stopping_checkbox, 
            n_beams_slider,
            no_repeat_ngram_size_slider],
        outputs = output_text
        )  

        
    demo.queue()
    demo.launch()