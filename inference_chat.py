from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import os
from eth_utils import is_address
import nacos
import logging
import time
from threading import Thread

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

max_seq_length = 2048
dtype = None
load_in_4bit = True

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Retrieve environment variables
model_name = os.getenv("MODEL_NAME")
wallet_address = os.getenv("WALLET_ADDRESS")
nacos_server = "nacos.hyperagi.network:80"  # Specify HTTPS and correct port
public_ip = os.getenv("PUBLIC_IP", "")
port = int(os.getenv("PORT", 5000))

# Check if model_name and wallet_address are not empty
if not model_name:
    raise ValueError("MODEL_NAME environment variable is not set or is empty")

if not wallet_address:
    raise ValueError("WALLET_ADDRESS environment variable is not set or is empty")

# Check if wallet_address is a valid Ethereum address
if not is_address(wallet_address):
    raise ValueError("WALLET_ADDRESS environment variable is not a valid Ethereum address")

if not public_ip:
    raise ValueError("PUBLIC_IP environment variable is not set or is empty")

# Initialize Nacos client
nacos_client = nacos.NacosClient(
    nacos_server,
    namespace="",
    username=os.getenv("NACOS_USERNAME", ""),
    password=os.getenv("NACOS_PASSWORD", "")
)

# Register service with Nacos
service_name = "hyperAGI-inference-chat"
ip = public_ip  # Use the actual public IP address
metadata = {"walletAddress": wallet_address}

# Log the parameters for debugging
logging.debug(f"Service Name: {service_name}")
logging.debug(f"IP: {ip}")
logging.debug(f"Port: {port}")
logging.debug(f"Metadata: {metadata}")

# Retry mechanism for registering with Nacos
max_retries = 5
for attempt in range(max_retries):
    try:
        response = nacos_client.add_naming_instance(service_name, ip, port, metadata=metadata)
        logging.info(f"Successfully registered with Nacos: {response}")
        break
    except Exception as e:
        logging.error(f"Failed to register with Nacos on attempt {attempt + 1}/{max_retries}: {e}")
        time.sleep(5)  # Wait for 5 seconds before retrying
else:
    raise RuntimeError("Failed to register with Nacos after several attempts")

# Heartbeat function
def send_heartbeat():
    while True:
        try:
            response = nacos_client.send_heartbeat(service_name, ip, port)
            logging.info(f"Heartbeat sent: {response}")
        except Exception as e:
            logging.error(f"Failed to send heartbeat: {e}")
        time.sleep(30)  # Send heartbeat every 30 seconds

# Start heartbeat thread
heartbeat_thread = Thread(target=send_heartbeat)
heartbeat_thread.daemon = True
heartbeat_thread.start()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    input_text = data.get("input_text")
    
    if not input_text:
        return jsonify({"error": "Please provide input_text"}), 400

    inputs = tokenizer(
        [alpaca_prompt.format('', input_text, "")],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, max_new_tokens=64)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part
    response_start = "### Response:\n"
    response = generated_text.split(response_start)[-1].strip()
    
    return jsonify({"generated_text": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
