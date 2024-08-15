from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
import os
from eth_utils import is_address
import nacos
import logging
import time
import queue
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

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
model, tokenizer = FastLanguageModel.from_pretrained(model_name, dtype=dtype, load_in_4bit=load_in_4bit)
FastLanguageModel.for_inference(model)

# Request handling with queue
request_queue = queue.Queue()
inference_lock = threading.Lock()

def batch_inference():
    while True:
        batch = []
        events = []
        start_time = time.time()

        while (time.time() - start_time) < INFER_TIMEOUT:
            try:
                req_data = request_queue.get(timeout=INFER_TIMEOUT - (time.time() - start_time))
                batch.append(req_data['data'])
                events.append(req_data['event'])
            except queue.Empty:
                if batch:
                    break

        if batch:
            with inference_lock:
                inputs = tokenizer([alpaca_prompt.format('Please generate a motion script according to the following description', text, "") for text in batch],
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   max_length=max_seq_length).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1, use_cache=True)
                responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

                for response, event in zip(responses, events):
                    event.response = response
                    event.num_input_tokens = len(tokenizer(event.data, return_tensors="pt").input_ids[0])
                    event.num_output_tokens = len(tokenizer(response, return_tensors="pt").input_ids[0])
                    event.set()

# Start batch processing thread
threading.Thread(target=batch_inference, daemon=True).start()

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    input_text = data.get("input_text")
    if not input_text:
        return jsonify({"error": "Please provide input_text"}), 400

    event = threading.Event()
    request_queue.put({'data': input_text, 'event': event})
    event.wait()

    # Extract the response part
    response_start = "### Response:\n"
    response = event.response.split(response_start)[-1].strip()

    return jsonify({
        "generated_text": response,
        "num_output_tokens": event.num_output_tokens,
        "num_input_tokens": event.num_input_tokens
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
