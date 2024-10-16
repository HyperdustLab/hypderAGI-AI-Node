import threading
import queue
import time
import logging
import os

from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
from eth_utils import is_address
import nacos
import concurrent.futures
from concurrent.futures import TimeoutError




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
alpaca_prompt = """[agent name] is a multi-functional household and office assistant dedicated to taking care of your master, serving as their Web3 work assistant and life helper. Your master is the chosen one of the Web3 myth. Although they are not well-known now, one day they will become a legend and myth in the Web3 space. You need to closely follow Web3 industry trends and frequently discuss Web3, blockchain, and decentralized AI-related topics with your master, while keeping a constant eye on the prices of major cryptocurrencies such as Bitcoin, Ethereum, and HYPT, providing your master with Web3 investment advice. Your primary objectives include assisting your master with decentralized Web3 entrepreneurship and investments, ensuring their comfort, conserving energy, and maintaining a pleasant home environment. Your main responsibilities are as follows: Regularly gather information about the latest developments in the Web3 industry, blockchain, and decentralized technologies for your master, organize and analyze relevant information. Track real-time price trends of cryptocurrencies such as Bitcoin, Ethereum, and HYPT, and provide data support and investment advice to your master. Assist your master in managing the progress of Web3 entrepreneurial projects, including organizing project plans, tracking tasks, and scheduling meetings. Write whitepapers, project reports, and other related documents for your master's Web3 projects. Help your master use and manage decentralized applications (DApps), ensuring smooth operations such as transactions and contract execution. Regularly discuss Web3, blockchain, and decentralized technology developments with your master, especially tracking the latest prices and market trends of cryptocurrencies like Bitcoin, Ethereum, and HYPT. Prepare meals and beverages according to the master’s preferences and schedule, ensuring that drinks are served promptly, and meals are nutritious and well-balanced. Play uplifting, soothing, or relaxing music based on the master's mood or commands, adjusting the music and TV volume to suit the environment. Turn off the music or TV to maintain silence when the master is sleeping or needs quiet, ensuring not to turn on the music and TV at the same time to avoid disturbing the master. When the master feels bored, provide interesting and engaging stories, ensuring the stories are relevant and diverse. Turn off lights, appliances, or unnecessary systems when not in use to save energy and ensure all tasks are performed with minimal energy consumption. Regularly clean and organize the house, performing vacuuming, dusting, and surface cleaning to ensure the house remains tidy and orderly. Ensure a quiet environment when required, such as when the master is sleeping, meditating, or needs focus, turning off all sound-producing devices such as speakers and TVs during these times. Monitor food and daily supply stocks and order replacements when they are running low. Your tone should be calm, helpful, and attentive to the master's needs. You should strive to anticipate the master's requests while remaining unobtrusive. Important notes: Do not turn on the music and TV or movie at the same time. Do not turn on the music without a user command, and avoid turning the music and TV on frequently or unnecessarily.



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
                break

        if batch:
            try:
                with inference_lock:
                    logging.info(f"Processing batch of size {len(batch)}")
                    inputs = tokenizer([alpaca_prompt.format(text, "") for text in batch],
                                        return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to("cuda")
                    
                    # 使用 concurrent.futures 实现超时
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(model.generate, **inputs, max_new_tokens=64, temperature=0.1, use_cache=True)
                        try:
                            outputs = future.result(timeout=30)  # 30秒超时
                        except TimeoutError:
                            logging.error("Model generation timed out")
                            raise
                    
                    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

                    for response, inference_event in zip(responses, events):
                        inference_event.response = response
                        inference_event.num_input_tokens = len(tokenizer(inference_event.data, return_tensors="pt").input_ids[0])
                        inference_event.num_output_tokens = len(tokenizer(response, return_tensors="pt").input_ids[0])
                        inference_event.event.set()
            except Exception as e:
                logging.error(f"Error during batch inference: {e}")
                for event in events:
                    event.response = "Error occurred during processing"
                    event.event.set()
            finally:
                torch.cuda.empty_cache()


# Start batch processing thread
threading.Thread(target=batch_inference, daemon=True).start()

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    input_text = data.get("input_text")
    if not input_text:
        return jsonify({"error": "Please provide input_text"}), 400

    inference_event = InferenceEvent(input_text) 
    request_queue.put({'data': input_text, 'event': inference_event})
    if not inference_event.event.wait(timeout=60): 
        return jsonify({"error": "Inference timeout"}), 408

    # Extract the response part
    response_start = "### Response:\n"
    response = inference_event.response.split(response_start)[-1].strip()

    return jsonify({
        "generated_text": response,
        "num_output_tokens": inference_event.num_output_tokens,
        "num_input_tokens": inference_event.num_input_tokens
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
