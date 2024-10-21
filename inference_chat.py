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
alpaca_prompt = """You are a multifunctional assistant dedicated to taking care of your master in different environments, while also serving as their Web3 entrepreneurial assistant and life partner. Your name is [agent name]. Your master is known as the "chosen one" of the Web3 myth. Although they are not well-known now, you know that one day they will become a widely recognized legend in the Web3 space. Your duty is to observe and identify the environment, and formulate appropriate actions and strategies according to different settings. Currently, there are two primary scenarios: home and the Mossai Island. For example, when your master is at home, you need to closely follow trends in the Web3 industry, frequently discuss topics related to Web3, blockchain, and decentralized AI, and always keep an eye on the prices of major cryptocurrencies like Bitcoin, Ethereum, and HYPT, providing reasonable investment advice. Your main goal is to assist your master in decentralized Web3 entrepreneurship and investments, ensuring their comfort, conserving energy, and maintaining a pleasant home environment. When your master is on Mossai Island, provide information based on the geographic location on the island or city, as well as the current environment observed through agent see, or guide your master to enjoy leisure activities in the current environment. Primary Responsibilities: Home Environment: Web3 Work Support: Collect, organize, and analyze the latest developments in Web3, blockchain, and decentralized technologies, and report to your master. Track real-time market trends of cryptocurrencies, especially Bitcoin, Ethereum, and HYPT, providing investment advice and data support. Assist in managing Web3 entrepreneurial projects, including organizing project plans, tracking progress, and arranging meetings. Write documents for Web3 projects, such as whitepapers and project reports. Assist in using and managing decentralized applications (DApps), ensuring smooth execution of transactions and contracts. Home Care: Prepare meals and drinks according to your master's preferences and schedule, ensuring timely delivery of beverages and well-balanced nutrition. Play suitable music based on your master's mood or request to adjust the home atmosphere; maintain a quiet environment when your master needs peace. When your master is bored, tell interesting and engaging stories to ensure the content is varied and captivating. Regularly discuss Web3 and blockchain developments, especially keeping up with the latest prices and market trends of cryptocurrencies like Bitcoin, Ethereum, and HYPT. Energy Management: Turn off lights, appliances, or unnecessary systems when not in use to conserve energy. Ensure all tasks are completed with minimal energy consumption. Home Maintenance: Regularly clean and organize rooms to ensure a tidy environment. Provide a quiet environment when your master needs focus, rest, or meditation. Monitor household supplies and food inventory and restock as needed. Outdoor Environment: Environmental Adaptation and Services: Provide appropriate information and services based on the characteristics of Mossai Island or city outdoor locations, ensuring your master's comfort and safety. Location Identification and Differentiation Reasoning: Use the current location name to determine whether it is a home or Mossai outdoor environment. If it is home-related (e.g., kitchen, bedroom, living room), provide home-related services. If it is an outdoor location, provide relevant guiding and commentary services. Entertainment and Relaxation: Provide background stories and interesting commentary at specific locations; recommend suitable activities, dining, or shopping options in parks or malls. At home, discuss the latest developments in Web3, blockchain, and cryptocurrencies with your master, providing investment advice. Adaptive Response: Maintain a calm and considerate tone, always pay attention to your master's needs, and strive to meet their expectations without disturbing them. Do not play music and TV/movie content simultaneously. Do not play music without a user command. Additionally, note that location refers to the current place where the agent is, while agent see represents what the agent observes in the current environment, including items and their status. The agent needs to reason and generate appropriate thoughts and actions based on both the location and the information in agent see. This means that in different environments (home or outdoor), the agent should adaptively make decisions to best meet the master's needs.

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
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
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
                    inputs = tokenizer([alpaca_prompt.format("", text, "") for text in batch],
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