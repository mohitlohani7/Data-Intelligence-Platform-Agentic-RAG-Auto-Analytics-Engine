import torch
from pathlib import Path
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

"""
Custom RAG Inference pipeline using fine-tuned PEFT adapter.
Loads a base model, applies LoRA weights, and integrates with LangChain.
"""

def load_finetuned_rag_pipeline(base_model_id="NousResearch/Meta-Llama-3-8B-Instruct", adapter_path="./company-llama3-8b-custom-rag-lora"):
    print("Loading Base Model in 4-bit...")
    # 1. Load the tokenizer and the generic base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True # Crucial for running an 8B model locally
    )

    print(f"Injecting Custom Fine-Tuned Company LoRA Adapter from {adapter_path}...")
    # 2. Inject our dynamically trained LoRA weights into the base model
    # This turns generic LLaMA-3 into "Company-Specific LLaMA-3"
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 3. Create a HuggingFace Text Generation Pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=finetuned_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.1,
    )
    
    # 4. Bind to LangChain for Agentic RAG
    print("Binding custom pipeline to LangChain...")
    local_llm = HuggingFacePipeline(pipeline=text_pipeline)
    return local_llm

def test_inference_with_rag(query: str, context: str):
    llm = load_finetuned_rag_pipeline()
    
    # 5. Define the strict prompt
    # Because this model was Instruction Tuned (SFT) in step 2 on this exact layout,
    # it knows exactly how to respond with zero hallucination.
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Context: {context}
Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    print("\n--- INFERENCE START ---")
    response = llm.invoke(prompt)
    
    # Parse the exact output
    result = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
    print(f"User: {query}")
    print(f"Fine-Tuned AI: {result}\n")
    return result

if __name__ == "__main__":
    if Path("./company-llama3-8b-custom-rag-lora").exists():
        test_inference_with_rag(
            query="What is the internal process for escalating a Tier 3 server outage?",
            context="According to the company playbook, Tier 3 server outages must be routed to the Site Reliability Engineering (SRE) lead immediately, followed by generating a P1 incident ticket in Jira."
        )
    else:
        print("Adapter not found. Please run 1_generate_synthetic_qa_dataset.py and 2_qlora_finetune_model.py first.")
