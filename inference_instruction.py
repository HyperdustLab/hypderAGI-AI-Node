import threading
import time
import logging
import os
import json
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
from eth_utils import is_address
import nacos
from peft import PeftModel
import subprocess
from pathlib import Path
import filelock
import shutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration and constants
max_seq_length = 2048
INFER_TIMEOUT = 5  # Timeout for inference batch processing
model_name = os.getenv("MODEL_NAME", "")
wallet_address = os.getenv("WALLET_ADDRESS", "")
nacos_server = os.getenv("NACOS_SERVER", "nacos.hyperagi.network:80")
public_ip = os.getenv("PUBLIC_IP", "")
port = int(os.getenv("PORT", 5000))
service_name = os.getenv("SERVICE_NAME", "hyperAGI-inference")
hf_token = os.getenv("HF_TOKEN", "")
dtype = None
load_in_4bit = True
base_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


# Path normalization (supports model name special characters) [2,5](@ref)
def sanitize_model_path(name):
    return Path("/models") / name.replace("/", "__").replace(" ", "_")

MODEL_CACHE_DIR = sanitize_model_path(model_name)
BASE_MODEL_CACHE_DIR = sanitize_model_path(base_model_name)

# Required file list (enhances completeness verification) [2,5](@ref)
REQUIRED_FILES = {
    "base": ["config.json", "model.safetensors", "tokenizer.json"],
    "adapter": ["adapter_config.json", "adapter_model.safetensors"]
}

def validate_model_files(model_dir, model_type):
    """Enhanced file validation (supports wildcard matching) [2,5](@ref)"""
    return all((model_dir / file).exists() for file in REQUIRED_FILES[model_type])




# Path normalization (supports model name special characters) [2,5](@ref)
def sanitize_model_path(name):
    return Path("/models") / name.replace("/", "__").replace(" ", "_")

MODEL_CACHE_DIR = sanitize_model_path(model_name)
BASE_MODEL_CACHE_DIR = sanitize_model_path(base_model_name)

# Required file list (enhances completeness verification) [2,5](@ref)
REQUIRED_FILES = {
    "base": ["config.json", "model.safetensors", "tokenizer.json"],
    "adapter": ["adapter_config.json", "adapter_model.safetensors"]
}

def validate_model_files(model_dir, model_type):
    """Enhanced file validation (supports wildcard matching) [2,5](@ref)"""
    return all((model_dir / file).exists() for file in REQUIRED_FILES[model_type])



def modify_adapter_config(model_dir):

    config_path = os.path.join(model_dir, "adapter_config.json")
    
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        original_path = config.get("base_model_name_or_path", "")
        config["base_model_name_or_path"] = BASE_MODEL_CACHE_DIR
        logging.info(f"Updated base_model path: {original_path} -> {BASE_MODEL_CACHE_DIR}")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        with open(config_path, "r") as f:
            verified_config = json.load(f)
            if verified_config["base_model_name_or_path"] != BASE_MODEL_CACHE_DIR:
                raise ValueError("Config modification failed")
                
        logging.info("Adapter config successfully modified")
        
    except Exception as e:
        logging.error(f"Failed to modify adapter config: {str(e)}")
        raise



# Prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}

### Response:
{}"""

# Validate environment variables
if not model_name:
    raise ValueError("MODEL_NAME environment variable is not set or is empty")
if not wallet_address or not is_address(wallet_address):
    raise ValueError("Invalid or empty WALLET_ADDRESS environment variable")
if not public_ip:
    raise ValueError("PUBLIC_IP environment variable is not set or is empty")

# Custom event class to encapsulate data and response
class InferenceEvent:
    def __init__(self, data):
        self.event = threading.Event()
        self.data = data
        self.response = None
        self.num_input_tokens = 0
        self.num_output_tokens = 0


# Heartbeat function with improved error handling
def send_heartbeat():
    while True:
        try:
            nacos_client.send_heartbeat(service_name, public_ip, port, metadata={"walletAddress": wallet_address})
            logging.info("Heartbeat sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send heartbeat: {e}")
        time.sleep(5)



# 带锁的智能下载（解决并发下载问题）[2,8](@ref)
def safe_download(model_dir, repo_id):
    lock_file = model_dir / ".download.lock"
    
    with filelock.FileLock(lock_file, timeout=600):
        if validate_model_files(model_dir, "base" if "Meta-Llama" in repo_id else "adapter"):
            logging.info(f"Model {repo_id} already exists")
            return

        logging.info(f"Starting download: {repo_id}")
        for attempt in range(3):
            try:
                subprocess.run([
                    "huggingface-cli", "download", repo_id,
                    "--local-dir", str(model_dir),
                    "--resume-download",
                    "--local-dir-use-symlinks", "True",
                    "--cache-dir", "/tmp/hf_cache"
                ], check=True)
                (model_dir / ".download_complete").touch()
                return
            except subprocess.CalledProcessError as e:
                logging.error(f"Download attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:
                    shutil.rmtree(model_dir, ignore_errors=True)
                    raise


# 重试装饰器（带自动清理）[2,5](@ref)
def retry_model_load(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args,**kwargs)
                except Exception as e:
                    logging.error(f"Model load attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                    shutil.rmtree(args[0], ignore_errors=True)
                    safe_download(*args, **kwargs)
            return None
        return wrapper
    return decorator

@retry_model_load()
def load_local_model():
    global model, tokenizer
    try:
        # 分步加载基础模型和适配器[2,5](@ref)
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=str(BASE_MODEL_CACHE_DIR),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        
        model = PeftModel.from_pretrained(
            base_model, 
            str(MODEL_CACHE_DIR),
            is_trainable=False
        )
        
        FastLanguageModel.for_inference(model)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Critical load failure: {str(e)}")
        raise


def initialize_models():
    """Enhanced initialization process (with auto-repair) [2,5](@ref)"""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BASE_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Base model check
        if not validate_model_files(BASE_MODEL_CACHE_DIR, "base"):
            safe_download(BASE_MODEL_CACHE_DIR, base_model_name)

        # Adapter model check
        if not validate_model_files(MODEL_CACHE_DIR, "adapter"):
            safe_download(MODEL_CACHE_DIR, model_name)

        modify_adapter_config(model_dir)  # 网页1
        load_local_model()

    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}")
        shutil.rmtree(MODEL_CACHE_DIR, ignore_errors=True)
        raise


# Nacos client setup
nacos_client = nacos.NacosClient(nacos_server, namespace="", username=os.getenv("NACOS_USERNAME", ""), password=os.getenv("NACOS_PASSWORD", ""))

# Service registration with retries
max_retries = 5
for attempt in range(max_retries):
    try:
        response = nacos_client.add_naming_instance(service_name, public_ip, port, metadata={"walletAddress": wallet_address})
        logging.info(f"Successfully registered with Nacos: {response}")
        break
    except Exception as e:
        logging.error(f"Failed to register with Nacos on attempt {attempt + 1}: {e}")
        time.sleep(5)
else:
    raise RuntimeError("Failed to register with Nacos after several attempts")




initialize_models()

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
heartbeat_thread.start()



def check_gpu_memory_usage():
    """Check if GPU memory usage exceeds 95%"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        memory_usage = allocated_memory / total_memory * 100
        
        logging.info(f"GPU memory usage: {memory_usage:.2f}%")
        
        if memory_usage > 95:
            logging.warning(f"GPU memory usage exceeds 95%: {memory_usage:.2f}%")
            return True
    return False


def clear_cuda_cache():
    """Clear the GPU memory cache."""
    logging.info("Clearing GPU memory cache.")
    torch.cuda.empty_cache()


@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Memory usage check (added threshold check)
        if check_gpu_memory_usage():
            return jsonify({"error": "GPU memory usage exceeds 95%, unable to process request"}), 503

        # Get request data
        data = request.json
        if not data or "input_text" not in data:
            return jsonify({"error": "Missing input_text parameter"}), 400

        input_text = data["input_text"]

        # Format input template
        formatted_input = alpaca_prompt.format(input_text, "")

        try:
            # Encode input
            model_input = tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length
            ).to(model.device)

            # Generate output
            outputs = model.generate(
                **model_input,
                max_new_tokens=64,
                temperature=0.1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            # Decode and process output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("\n")[-1].strip()

            # Count token usage
            num_input_tokens = len(outputs[0]) - (outputs[0] == tokenizer.pad_token_id).sum().item()
            num_output_tokens = len(outputs[0])

            logging.info(f"Processed {num_input_tokens} input tokens, generated {num_output_tokens} output tokens")

            return jsonify({
                "generated_text": response,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens
            })

        except RuntimeError as e:
            logging.error(f"CUDA memory error: {str(e)}")
            return jsonify({"error": "Inference failed due to memory constraints"}), 500
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            # Safe memory release
            if 'model_input' in locals():
                del model_input
            if 'outputs' in locals():
                del outputs
            clear_cuda_cache()

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)