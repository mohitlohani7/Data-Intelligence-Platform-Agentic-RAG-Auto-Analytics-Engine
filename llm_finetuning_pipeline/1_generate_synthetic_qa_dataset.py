import os
import json
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tqdm import tqdm

"""
Script to extract raw text from PDFs and build an instruction-tuning dataset.
Uses a teacher model to generate QA pairs.
"""

load_dotenv()

# Initialize Teacher Model (e.g., Groq Llama3-70b or OpenAI GPT-4o)
# We use a large, smart model to teach our smaller local open-source model.
teacher_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)

dataset_prompt = PromptTemplate.from_template("""
You are an expert data curator for an enterprise AI team.
Given the following raw text chunk from a company proprietary document, generate 3 highly diverse, complex Question-Answer pairs that a senior employee might ask.
Format the output EXACTLY as valid JSON array of objects with keys: "instruction", "input" (the context), and "output" (the precise answer).

Raw Text Chunk: 
{context}

JSON Output:
""")

def extract_chunks_from_pdfs(pdf_dir="data/pdfs"):
    print(f"Extracting chunks from PDFs in {pdf_dir}...")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created {pdf_dir}. Please place your PDFs there and rerun.")
        return []

    text = ""
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, file))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return splitter.split_text(text)

def generate_dataset():
    chunks = extract_chunks_from_pdfs()
    if not chunks: return

    dataset = []
    print(f"Generating synthetic instruction dataset from {len(chunks)} chunks...")
    
    for chunk in tqdm(chunks[:50]):  # Limit to 50 chunks for demonstration
        try:
            response = teacher_llm.invoke(dataset_prompt.format(context=chunk))
            # Clean LLM output to extract pure JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            qa_pairs = json.loads(content)
            dataset.extend(qa_pairs)
        except Exception as e:
            print(f"Failed to generate for a chunk: {str(e)}")
            continue

    output_file = "company_finetuning_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"\n✅ Successfully generated {len(dataset)} training examples!")
    print(f"Dataset saved to {output_file} in Alpaca format (Ready for QLoRA).")

if __name__ == "__main__":
    generate_dataset()
