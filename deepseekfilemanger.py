# deepseek_local_analyzer.py

import utils.transport
import utils.session
import os
import time
import warnings
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from datetime import datetime
import zipfile
import tempfile
import shutil
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit.components.v1 as components

# ----------------------------- Configuration -----------------------------
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
CACHE_DIR = "./model_cache"
MAX_CHUNK_LINES = 1000
MAX_CONTEXT_LENGTH = 4096
DEFAULT_MAX_WORKERS = 3
EXCLUDE_DIRS = {'node_modules', 'venv', '__pycache__', '.git'}
OUTPUT_REPORT_MD = "deepseek_analysis.md"
OUTPUT_REPORT_JSON = "deepseek_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        RotatingFileHandler("analysis.log", maxBytes=1e6, backupCount=3)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------- Model Loading ------------------------------
@st.cache_resource
def load_deepseek_model():
    """Load DeepSeek model with optimized settings"""
    try:
        logger.info("Loading model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            use_fast=True,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        if DEVICE == "cpu":
            model = model.to(DEVICE)
            model = model.float()

        logger.info(f"Model loaded on {model.device}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"Model loading failed: {str(e)}")
        raise

# ---------------------------- Core Functions -----------------------------
def analyze_code_chunk(model, tokenizer, code_chunk: str) -> str:
    """Analyze code chunk using DeepSeek model"""
    try:
        prompt = (
            "You are an expert code analyst. Analyze this code for:\n"
            "- Bugs and security issues\n- Code smells\n- Optimizations\n"
            "- Architectural improvements\n\n"
            f"Code:\n```\n{code_chunk}\n```\n\nAnalysis:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_CONTEXT_LENGTH,
            truncation=True
        ).to(DEVICE)

        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            do_sample=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return f"Analysis error: {str(e)}"

# ---------------------------- File Processing ----------------------------
def process_file(file_path: Path, model, tokenizer, progress_callback) -> dict:
    """Process a single file with DeepSeek analysis"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return {"file_path": str(file_path), "analysis": "**Binary File** Skipped"}

    chunks = [content[i:i+MAX_CHUNK_LINES] for i in range(0, len(content), MAX_CHUNK_LINES)]
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

# ---------------------------- UI Components ------------------------------
def main():
    st.set_page_config(page_title="DeepSeek Code Analyzer", layout="wide")
    st.title("🔍 Local DeepSeek Code Analyzer")

    # Load model once
    try:
        model, tokenizer = load_deepseek_model()
    except:
        st.stop()

    # Sidebar Configuration
    st.sidebar.header("Settings")
    uploaded_zip = st.sidebar.file_uploader("Upload Codebase ZIP", type=["zip"])
    root_dir = handle_file_upload(uploaded_zip)
    output_format = st.sidebar.selectbox("Output Format", ["Markdown", "JSON"], index=0)
    output_file = st.sidebar.text_input("Output File", OUTPUT_REPORT_MD)
    max_workers = st.sidebar.slider("Parallel Workers", 1, 5, DEFAULT_MAX_WORKERS)

    # Main Analysis
    if st.sidebar.button("Start Analysis"):
        with st.spinner("Analyzing codebase..."):
            try:
                report = process_codebase(root_dir, model, tokenizer, output_file, max_workers, output_format)
                display_results(report, output_file, output_format)
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
        if not Path(default_dir).is_dir():
            st.error("Invalid directory path")
            st.stop()
        return default_dir

def process_codebase(root_dir, model, tokenizer, output_file, max_workers, output_format):
    """Main processing pipeline"""
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
    """Generate output report"""
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
    """Display results and download options"""
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
    """Get all files in directory with exclusions"""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            all_files.append(Path(dirpath) / f)
    return all_files

if __name__ == "__main__":
    main()