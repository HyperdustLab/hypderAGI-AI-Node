from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import os
from eth_utils import is_address
import nacos
import logging
import time
import queue
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import TextStreamer

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuration and constant
max_seq_length = 2048
batch_time_limit = 2  # 5秒的请求收集时间
model_name = os.getenv("MODEL_NAME", "")
wallet_address = os.getenv("WALLET_ADDRESS", "")
nacos_server = os.getenv("NACOS_SERVER", "nacos.hyperagi.network:80")
public_ip = os.getenv("PUBLIC_IP", "")
port = int(os.getenv("PORT", 5000))
service_name = os.getenv("SERVICE_NAME", "hyperAGI-inference-chat")
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
adapter_name = model_name
logging.info(f'Model name: {adapter_name}')

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 应用PEFT适配器
model = PeftModel.from_pretrained(model, adapter_name)
FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

# Request handling with queue and batching
request_queue = queue.Queue()
inference_lock = threading.Lock()

def validate_and_clean_probs(probs):
    """Ensure probabilities are valid by handling NaN, Inf, and negative values."""
    if torch.any(torch.isnan(probs)):
        logging.error("Probability tensor contains NaN values.")
        probs = torch.nan_to_num(probs, nan=0.0)  # 将 NaN 转换为 0
    if torch.any(torch.isinf(probs)):
        logging.error("Probability tensor contains Inf values.")
        probs = torch.clamp(probs, max=1.0)  # 将 Inf 限制为 1.0
    if torch.any(probs < 0):
        logging.error("Probability tensor contains negative values.")
        probs = torch.clamp(probs, min=0.0)  # 将负值限制为 0
    return probs

def batch_inference():
    while True:
        with inference_lock:
            batch = []
            events = []
            start_time = time.time()

            while time.time() - start_time < batch_time_limit:
                try:
                    req_data = request_queue.get(timeout=1)
                    batch.append(req_data['data'])
                    events.append(req_data['event'])
                except queue.Empty:
                    continue

            if batch:
                logging.info(f"Preparing inputs for {len(batch)} requests.")
                try:
                    # 准备输入数据
                    inputs = tokenizer([alpaca_prompt.format('.', text, "") for text in batch],
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       max_length=max_seq_length).to("cuda")
                    logging.info("Inputs prepared successfully.")

                    # 开始生成输出
                    inference_start_time = time.time()
                    logging.info("Generating outputs...")

                    # 模型生成
                    outputs = model.generate(**inputs, max_new_tokens=64)
                    inference_end_time = time.time()
                    logging.info(f"Outputs generated in {inference_end_time - inference_start_time:.4f} seconds.")

                    # 解码响应
                    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
                    logging.info("Responses generated.")

                    # 记录处理每个响应的时间
                    processing_start_time = time.time()
                    for response, event, input_text in zip(responses, events, batch):
                        event.response = response
                        event.num_input_tokens = len(tokenizer(input_text, return_tensors="pt").input_ids[0])
                        event.num_output_tokens = len(tokenizer(response, return_tensors="pt").input_ids[0])
                        event.set()
                    processing_end_time = time.time()
                    logging.info(f"Processed responses in {processing_end_time - processing_start_time:.4f} seconds.")
                except Exception as e:
                    logging.error(f"Error during batch inference: {e}", exc_info=True)


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

    response_start = "### Response:\n"
    response = event.response.split(response_start)[-1].strip()

    return jsonify({
        "generated_text": response,
        "num_output_tokens": event.num_output_tokens,
        "num_input_tokens": event.num_input_tokens
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
