# deepseek_code_analyzer.py

import os
import warnings
import json
import logging
import tempfile
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging.handlers import RotatingFileHandler

# ---------------------------- Configuration -----------------------------
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
CACHE_DIR = "./model_cache"
MAX_CHUNK_SIZE = 1000  # Lines per chunk
MAX_CONTEXT_LENGTH = 4096
DEFAULT_WORKERS = 3
EXCLUDE_DIRS = {'node_modules', 'venv', '__pycache__', '.git'}
OUTPUT_REPORT_MD = "deepseek_analysis.md"
OUTPUT_REPORT_JSON = "deepseek_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added")
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("analysis.log", maxBytes=1e6, backupCount=3)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------- Model Loading -----------------------------
@st.cache_resource
def load_model():
    """Load and configure DeepSeek model with proper tokenization"""
    try:
        logger.info("Initializing model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            use_fast=True,
            padding_side='left',
            trust_remote_code=True
        )

        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        if DEVICE == "cpu":
            model = model.float()

        logger.info(f"Model loaded successfully on {model.device}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        st.error(f"Model loading error: {str(e)}")
        raise

# ---------------------------- Core Analysis -----------------------------
def analyze_code_chunk(model, tokenizer, code: str) -> str:
    """Perform analysis on a code chunk with proper attention handling"""
    try:
        prompt = (
            "As a senior engineer, analyze this code for:\n"
            "1. Potential bugs\n2. Security issues\n3. Optimizations\n"
            "4. Code smells\n5. Architectural improvements\n\n"
            f"Code:\n```\n{code}\n```\n\nAnalysis:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_CONTEXT_LENGTH,
            truncation=True,
            padding=True,
            return_attention_mask=True
        ).to(DEVICE)

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            do_sample=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Analysis error: {str(e)}"

# ---------------------------- File Processing ---------------------------
def process_file(file_path: Path, model, tokenizer, progress_callback) -> dict:
    """Process individual files with chunked analysis"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return {"file_path": str(file_path), "analysis": "**Binary File** Skipped"}

    chunks = [content[i:i+MAX_CHUNK_SIZE] for i in range(0, len(content), MAX_CHUNK_SIZE)]
    analyses = []
    
    for idx, chunk in enumerate(chunks, 1):
        analysis = analyze_code_chunk(model, tokenizer, chunk)
        analyses.append(f"### Chunk {idx} Analysis\n{analysis}")
        if progress_callback:
            progress_callback(file_path)

    return {
        "file_path": str(file_path),
        "analysis": "\n\n".join(analyses)
    }

# ---------------------------- UI Components -----------------------------
def main():
    """Main Streamlit application interface"""
    st.set_page_config(page_title="DeepSeek Code Analyzer", layout="wide")
    st.title("🔍 AI Codebase Analyzer")

    # Load model
    try:
        model, tokenizer = load_model()
    except:
        st.stop()

    # Sidebar configuration
    st.sidebar.header("Configuration")
    uploaded_zip = st.sidebar.file_uploader("Upload ZIP Archive", type=["zip"])
    root_dir = handle_file_upload(uploaded_zip)
    output_format = st.sidebar.selectbox("Report Format", ["Markdown", "JSON"])
    output_file = st.sidebar.text_input("Output File", OUTPUT_REPORT_MD)
    max_workers = st.sidebar.slider("Parallel Workers", 1, 5, DEFAULT_WORKERS)

    # Main analysis workflow
    if st.sidebar.button("Start Analysis"):
        with st.spinner("Analyzing codebase..."):
            try:
                report_data = process_codebase(
                    root_dir, 
                    model, 
                    tokenizer, 
                    output_file, 
                    max_workers, 
                    output_format
                )
                display_results(report_data, output_file, output_format)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# ---------------------------- Helper Functions ---------------------------
def handle_file_upload(uploaded_zip):
    """Handle file uploads and directory selection"""
    if uploaded_zip:
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(uploaded_zip) as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir
        except Exception as e:
            st.error(f"Invalid ZIP file: {str(e)}")
            st.stop()
    else:
        default_dir = st.sidebar.text_input("Project Directory", os.getcwd())
        if Path(default_dir).is_dir():
            return default_dir
        st.error("Invalid directory path")
        st.stop()

def process_codebase(root_dir, model, tokenizer, output_file, max_workers, output_format):
    """Orchestrate codebase analysis"""
    all_files = get_all_files(root_dir)
    if not all_files:
        st.warning("No files found for analysis")
        return []

    report_data = []
    progress_lock = Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        progress_bar = st.progress(0)
        total_files = len(all_files)
        completed = 0

        def update_progress(_):
            nonlocal completed
            with progress_lock:
                completed += 1
                progress_bar.progress(completed / total_files)

        for file in all_files:
            futures.append(
                executor.submit(
                    process_file,
                    file, model, tokenizer, update_progress
                )
            )

        for future in as_completed(futures):
            try:
                report_data.append(future.result())
            except Exception as e:
                logger.error(f"Processing failed: {e}")

    generate_report(report_data, output_file, output_format)
    return report_data

def generate_report(data, filename, format):
    """Generate analysis report"""
    try:
        if format == "Markdown":
            with open(filename, 'w') as f:
                f.write("# DeepSeek Code Analysis Report\n\n")
                for item in data:
                    f.write(f"## {item['file_path']}\n\n{item['analysis']}\n\n")
        else:
            with open(filename, 'w') as f:
                json.dump({"analysis": data}, f, indent=2)
        st.success(f"Report generated: {filename}")
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")

def display_results(report, filename, format):
    """Display results with download option"""
    st.subheader("Analysis Results")
    
    if format == "Markdown":
        with open(filename) as f:
            st.markdown(f.read())
    else:
        with open(filename) as f:
            st.json(json.load(f))
    
    with open(filename, "rb") as f:
        st.download_button(
            "Download Report",
            f.read(),
            filename,
            "text/markdown" if format == "Markdown" else "application/json"
        )

def get_all_files(root_dir):
    """Retrieve files from directory"""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            all_files.append(Path(dirpath) / f)
    return all_files

if __name__ == "__main__":
    main()