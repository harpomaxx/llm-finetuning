from transformers import (AutoModelForCausalLM, AutoTokenizer)

model_code = AutoModelForCausalLM.from_pretrained(
        "../../../models/opt350m-codealpaca-20k/",
        local_files_only= True,
        #quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True)

model_code.push_to_hub("opt350m-codealpaca20k")
tokenizer.push_to_hub("opt350m-codealpaca20k")