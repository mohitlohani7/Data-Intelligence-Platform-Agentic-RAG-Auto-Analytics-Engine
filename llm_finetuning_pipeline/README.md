# 🎓 QLoRA Fine-Tuning Pipeline

This directory contains scripts to generate custom data from unstructured PDFs and parameter-efficiently fine-tune a LLaMA model natively using QLoRA. This pipeline solves the issue of relying solely on baseline instruction weights when encountering proprietary terminology or out-of-domain conversational queries.

## Pipeline Architecture

1. **`1_generate_synthetic_qa_dataset.py`**
   * Creates an "Instruction-Tuning" QA dataset automatically.
   * Extracts text from PDFs and leverages a large Teacher model (via Groq/OpenAI) to generate synthetically complex question-answer mappings. Output is `company_finetuning_dataset.json`.

2. **`2_qlora_finetune_model.py`**
   * Uses HuggingFace `SFTTrainer`, `peft`, and `bitsandbytes`.
   * Loads an open-weight base model (like `NousResearch/Meta-Llama-3-8B-Instruct`) in nf4 4-bit precision.
   * Attaches a LoRA adapter (rank 64) over attention modules to train strictly on the generated JSON dataset, reducing VRAM usage significantly.

3. **`3_custom_model_rag_inference.py`**
   * Test inference script. Loads the newly fine-tuned LoRA weights (`company-llama3-8b-custom-rag-lora`) back into a LangChain pipeline, demonstrating private RAG via the `HuggingFacePipeline` class.

---
*Ensure CUDA dependencies are configured appropriately before running the training script.*
