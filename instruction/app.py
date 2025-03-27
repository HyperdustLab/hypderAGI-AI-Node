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

MODEL_CACHE_DIR = "/models/HyperdustProtocol/HyperAuto-cog-llama3-8b-3407"
BASE_MODEL_CACHE_DIR = "/models/unsloth_Meta-Llama-3.1-8B-bnb-4bit"





def modify_adapter_config(model_dir):
    config_path = os.path.join(model_dir, "adapter_config.json")
    
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        original_path = config.get("base_model_name_or_path", "")
        config["base_model_name_or_path"] = str(BASE_MODEL_CACHE_DIR)
        logging.info(f"Updated base_model path: {original_path} -> {BASE_MODEL_CACHE_DIR}")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        with open(config_path, "r") as f:
            verified_config = json.load(f)
            if verified_config["base_model_name_or_path"] != str(BASE_MODEL_CACHE_DIR):
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

def load_local_model():
    global model, tokenizer
    try:
        # Step-by-step loading of base model and adapter [2,5](@ref)
        base_model,tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(BASE_MODEL_CACHE_DIR),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        
        model = PeftModel.from_pretrained(
            base_model, 
            MODEL_CACHE_DIR,
            is_trainable=False
        )
        
        FastLanguageModel.for_inference(model)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Critical load failure: {str(e)}")
        raise


def initialize_models():
    try:
        modify_adapter_config(MODEL_CACHE_DIR) 
        load_local_model()

    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}")
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