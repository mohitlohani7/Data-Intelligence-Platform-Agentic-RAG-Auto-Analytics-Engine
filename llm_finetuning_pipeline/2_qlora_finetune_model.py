import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

"""
QLoRA fine-tuning script.
Loads a base model in 4-bit and trains a LoRA adapter on custom Q&A dataset.
"""

# 1. HuggingFace Model (Non-gated equivalent of Llama-3-8B-Instruct for smooth execution)
model_name = "NousResearch/Meta-Llama-3-8B-Instruct" 
# Output folder for the new fine-tuned weights
new_lora_adapter = "company-llama3-8b-custom-rag-lora" 

# 2. BitsAndBytes 4-bit Quantization Config
# Fits a 15GB model into just 6GB VRAM for standard GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def format_instruction(example):
    """Formats the JSON into a training prompt understandable by the LLM."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Context: {example['input']}
Question: {example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""

# 3. Load Dataset
print("Loading fine-tuning dataset...")
dataset = load_dataset("json", data_files="company_finetuning_dataset.json", split="train")
dataset = dataset.map(lambda x: {"text": format_instruction(x)})

# 4. Load Base Model and Tokenizer
print(f"Loading base model {model_name} in 4-bit QLoRA...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # Auto maps to GPU if available
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model = prepare_model_for_kbit_training(base_model)

# 5. Define LoRA (Low-Rank Adaptation) Configuration
# This focuses the training ONLY on Attention modules, making it 90% cheaper and faster
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64, # High rank for maximum learning capability on dense company data
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
base_model = get_peft_model(base_model, peft_config)
base_model.print_trainable_parameters()

# 6. Setup Training Hyperparameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # Use bfloat16 for newer Ampere GPUs
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

# 7. Supervised Fine-Tuning (SFT) Trainer
print("🚀 Starting Industry-Grade QLoRA Fine-Tuning Pipeline...")
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Train the model
trainer.train()

# 8. Save the newly fine-tuned LoRA Adapter
trainer.model.save_pretrained(new_lora_adapter)
tokenizer.save_pretrained(new_lora_adapter)

print(f"✅ Successfully fine-tuned! Adapter weights saved to: ./{new_lora_adapter}")
print("You can now load this adapter into LangChain or Streamlit for an absolutely unmatched, domain-specific RAG system.")
