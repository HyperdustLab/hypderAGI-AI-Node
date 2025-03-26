import threading
import time
import logging
import os

from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
from eth_utils import is_address
import nacos
from peft import PeftModel
import concurrent.futures
from concurrent.futures import TimeoutError
from huggingface_hub import snapshot_download

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
dtype = None
load_in_4bit = True

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

# Heartbeat function with improved error handling
def send_heartbeat():
    while True:
        try:
            nacos_client.send_heartbeat(service_name, public_ip, port, metadata={"walletAddress": wallet_address})
            logging.info("Heartbeat sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send heartbeat: {e}")
        time.sleep(5)



# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model_dir = "/root/.cache/huggingface/"+model_name
try:
    snapshot_download(repo_id=model_name, local_dir=local_model_dir)
    # Add HuggingFace token configuration
    hf_token = os.getenv("HF_TOKEN", "")  # Get token from environment variable
    if not hf_token:
        logging.warning("HF_TOKEN not set. Attempting to load model without token.")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        local_model_dir,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token
    )
    FastLanguageModel.for_inference(model)

    logging.info("Model loaded and ready for inference.")

except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")


FastLanguageModel.for_inference(model)


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
        # Check GPU memory usage before processing request
        if check_gpu_memory_usage():
            return jsonify({"error": "GPU memory usage exceeds 95%, unable to process the request"}), 503
        
        # Get request data
        data = request.json
        input_text = data.get("input_text")
        if not input_text:
            return jsonify({"error": "Please provide input_text"}), 400

        inference_event = InferenceEvent(input_text)

        # Format the input based on the template
        formatted_input = alpaca_prompt.format(input_text, "")
        
        # Tokenize input and process it using the model
        model_input = tokenizer(formatted_input, return_tensors="pt").to(model.device)
        
        # Generate output with safe handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # 确保输入数据在正确的设备上
                for key in model_input:
                    if isinstance(model_input[key], torch.Tensor):
                        model_input[key] = model_input[key].to(model.device)
                
                # 修改生成参数
                future = executor.submit(
                    model.generate,
                    **model_input,
                    max_new_tokens=512,
                    temperature=0.1,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False  # 改为 False 以避免采样导致的问题
                )
                outputs = future.result(timeout=30)
                
            except Exception as e:
                logging.error(f"Generation error: {str(e)}")
                clear_cuda_cache()
                raise
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("\n")[-1].strip()
        
        # Calculate number of input tokens and output tokens
        num_input_tokens = len(model_input['input_ids'][0])
        num_output_tokens = len(outputs[0])

        # Release GPU memory after inference
        clear_cuda_cache()

        return jsonify({
            "generated_text": response,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens
        })
    
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)