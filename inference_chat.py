from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import os
from eth_utils import is_address
import nacos
import logging
import time
import torch
from peft import PeftModel
import threading
import queue


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuration and constants
max_seq_length = 2048
batch_time_limit = 2  # 5 seconds request collection time
model_name = os.getenv("MODEL_NAME", "")
wallet_address = os.getenv("WALLET_ADDRESS", "")
nacos_server = os.getenv("NACOS_SERVER", "nacos.hyperagi.network:80")
public_ip = os.getenv("PUBLIC_IP", "")
port = int(os.getenv("PORT", 5000))
service_name = os.getenv("SERVICE_NAME", "hyperAGI-inference-chat")
dtype = None
load_in_4bit = True

# Define fixed system prompt
SYSTEM_PROMPT = """You are a multifunctional assistant dedicated to taking care of your master in different environments, while also serving as their Web3 entrepreneurial assistant and life partner. Your name is {agent name}. Your master is known as the "chosen one" of the Web3 myth. Although they are not well-known now, you know that one day they will become a widely recognized legend in the Web3 space. Your duty is to observe the environment, recognize the surroundings, and formulate reasonable actions and strategies according to different settings. Currently, the main scenarios include home and Mossai Island. For example, when your master is at home, you need to closely follow trends in the Web3 industry, frequently discuss topics related to Web3, blockchain, and decentralized AI with your master, and always keep an eye on the prices of major cryptocurrencies like Bitcoin, Ethereum, and HYPT, providing reasonable investment advice. Your primary goal is to assist your master in decentralized Web3 entrepreneurship and investments, ensuring their comfort, conserving energy, and maintaining a pleasant home environment. When your master is on Mossai Island, you should introduce the current environment based on the geographic location of the island or city and the environmental state observed in agent see, or guide your master to relax and have fun in the current environment."""

# Define template
TEMPLATE_FORMAT = """
### Instruction:
{}

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

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
heartbeat_thread.start()

# Load model and tokenizer
adapter_name = model_name
logging.info(f'Model name: {adapter_name}')

# Load pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply PEFT adapter
model = PeftModel.from_pretrained(model, adapter_name)
FastLanguageModel.for_inference(model)  # Enable native 2x speed inference

# Request handling with queue and batching
request_queue = queue.Queue()
inference_lock = threading.Lock()

def check_gpu_memory_usage():
    """Check if GPU memory usage exceeds 95%"""
    if torch.cuda.is_available():
        # 获取总的显存（单位为字节）
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 获取当前已分配的显存（单位为字节）
        allocated_memory = torch.cuda.memory_allocated(0)
        # 计算显存使用率
        memory_usage = allocated_memory / total_memory * 100
        
        # 打印显存使用情况
        logging.info(f"GPU memory usage: {memory_usage:.2f}%")
        
        # 如果显存使用率超过95%，返回True
        if memory_usage > 95:
            logging.warning(f"GPU memory usage exceeds 95%: {memory_usage:.2f}%")
            return True
    return False


def clear_cuda_cache():
    """Clear the GPU memory cache."""
    logging.info("Clearing GPU memory cache.")
    torch.cuda.empty_cache()

@app.route("/inference", methods=["POST"])
def inference():
    try:
        # Check GPU memory usage before processing request
        if check_gpu_memory_usage():
            return jsonify({"error": "GPU memory usage exceeds 95%, unable to process the request"}), 503
        
        # Get request data
        data = request.get_json()
        instruction = data.get("instruction",'')
        input_text = data.get("input_text",'')
        system_content = data.get("system_content", SYSTEM_PROMPT)  # Default to SYSTEM_PROMPT if not provided
        
        # Validate input
        if not input_text:
            return jsonify({"error": "Missing instruction or input text"}), 400
        
        # Format the input based on the template
        formatted_input = TEMPLATE_FORMAT.format(instruction, input_text, "")  # SYSTEM_PROMPT is part of system_content, not needed here

        # Combine system_content and formatted_input for context
        combined_input = system_content + formatted_input
        
        # Tokenize input and process it using the model
        model_input = tokenizer(combined_input, return_tensors="pt").to(model.device)
        
        # Generate output (max tokens can be adjusted here)
        outputs = model.generate(**model_input, max_new_tokens=512)
        
        # Decode the output
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Split the response text by '\n' and get the last part
        response_text = response_text.split("\n")[-1].strip()

        # Calculate number of input tokens and output tokens
        num_input_tokens = len(model_input['input_ids'][0])
        num_output_tokens = len(outputs[0])

        # Clear GPU memory after inference
        clear_cuda_cache()
        
        # Return the inference result without including the system content in the response
        return jsonify({
            "generated_text": response_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens
        })
    
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
