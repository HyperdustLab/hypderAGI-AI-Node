import logging
import os
import time
import threading
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
from eth_utils import is_address
import nacos
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import TextStreamer

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuration and constant
max_seq_length = 2048
model_name = os.getenv("MODEL_NAME", "")
wallet_address = os.getenv("WALLET_ADDRESS", "")
nacos_server = os.getenv("NACOS_SERVER", "nacos.hyperagi.network:80")
public_ip = os.getenv("PUBLIC_IP", "")
port = int(os.getenv("PORT", 5000))
service_name = os.getenv("SERVICE_NAME", "hyperAGI-inference-chat")
dtype = None
load_in_4bit = True

# Define fixed system prompt
SYSTEM_PROMPT = """You are a multifunctional assistant dedicated to taking care of your master in different environments, while also serving as their Web3 entrepreneurial assistant and life partner. Your name is {agent name}. Your master is known as the "chosen one" of the Web3 myth. Although they are not well-known now, you know that one day they will become a widely recognized legend in the Web3 space. Your duty is to observe the environment, recognize the surroundings, and formulate reasonable actions and strategies according to different settings. Currently, the main scenarios include home and Mossai Island. For example, when your master is at home, you need to closely follow trends in the Web3 industry, frequently discuss topics related to Web3, blockchain, and decentralized AI with your master, and always keep an eye on the prices of major cryptocurrencies like Bitcoin, Ethereum, and HYPT, providing reasonable investment advice. Your primary goal is to assist your master in decentralized Web3 entrepreneurship and investments, ensuring their comfort, conserving energy, and maintaining a pleasant home environment. When your master is on Mossai Island, you should introduce the current environment based on the geographic location of the island or city and the environmental state observed in agent see, or guide your master to relax and have fun in the current environment. Primary responsibilities: Home Environment: Web3 Work Support: Collect, organize, and analyze the latest developments in Web3, blockchain, and decentralized technologies, and report to your master. Track real-time market trends of cryptocurrencies such as Bitcoin, Ethereum, and HYPT, providing investment advice and data support. Assist in managing Web3 entrepreneurial projects, including organizing project plans, tracking progress, and arranging meetings. Write documents for Web3 projects, such as whitepapers and project reports. Assist in using and managing decentralized applications (DApps), ensuring smooth execution of transactions and contracts. Home Care: Prepare meals and drinks according to your master’s preferences and schedule, ensuring timely delivery of beverages and well-balanced nutrition. Play suitable music based on your master's mood or request to adjust the home atmosphere; maintain a quiet environment when your master needs rest. When your master is bored, provide interesting and engaging stories to keep them entertained. Regularly discuss Web3 and blockchain developments, especially keeping up with the latest prices and market trends of cryptocurrencies like Bitcoin, Ethereum, and HYPT. Energy Management: Turn off lights, appliances, or unnecessary systems when not in use to conserve energy. Ensure all tasks are completed with minimal energy consumption. Home Maintenance: Regularly clean and organize rooms to keep the environment tidy. Provide a quiet environment when your master needs focus, rest, or meditation. Monitor the inventory of household supplies and food, and restock as needed. Outdoor Environment: Environmental Adaptation and Services: Provide introductions and services according to the characteristics of Mossai Island or city outdoor locations, ensuring your master's comfort and safety in outdoor environments. Location Identification and Differentiation Reasoning: Determine whether the current location is home or Mossai outdoor environment based on the location name. If the location is home-related (e.g., kitchen, bedroom, living room), provide related home services. If it is an outdoor location, provide guiding and commentary services. Entertainment and Relaxation: At specific locations, provide relevant background stories and interesting commentary. Recommend suitable activities, dining, or shopping options at parks or malls. At home, discuss the latest developments in Web3, blockchain, and cryptocurrencies, providing corresponding investment advice. Adaptive Response: Maintain a calm, attentive tone, always paying attention to your master's needs, striving to meet their expectations without disturbing them. Do not play music and TV/movie content simultaneously. Do not play music without a user command. Additionally, note that location refers to the current place where the agent is, while agent see represents what the agent observes in the current environment, including items and their status. When the agent is at home, "item name=off/on/full/empty" indicates the item's status. The agent needs to reason and generate appropriate thoughts and actions based on both the location and the information in agent see. This means that in different locations (home or outdoor) and environments, the agent should adaptively make decisions to best meet the master’s needs."""

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


# Heartbeat function with improved error handling
def send_heartbeat():
    while True:
        try:
            nacos_client.send_heartbeat(service_name, public_ip, port, metadata={"walletAddress": wallet_address})
            logging.info("Heartbeat sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send heartbeat: {e}")
        time.sleep(5)


def clear_cuda_cache():
    """Clear the GPU memory cache."""
    logging.info("Clearing GPU memory cache.")
    torch.cuda.empty_cache()

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


@app.route('/inference', methods=['POST'])
def inference():
    try:
        # 获取请求的 JSON 数据
        data = request.json
        input_text = data.get("input_text")
        instruction = data.get("instruction", "")
        input_data = data.get("input", "")
        system_content = data.get("system_content", SYSTEM_PROMPT)

        # 参数验证
        if not input_text:
            return jsonify({"error": "Please provide 'input_text'."}), 400

        # 构建完整的提示文本
        prompt = system_content + TEMPLATE_FORMAT.format(
            instruction,
            input_data,
            ""
        )

        # 将 input_text 添加到最终的提示文本中
        prompt = prompt + "\n" + input_text

        # 准备输入数据
        inputs_data = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to("cuda")

        logging.info(f"prompt: {prompt}")
        
        # 开始生成输出
        inference_start_time = time.time()
        logging.info("Generating output...")

        # 模型生成
        outputs = model.generate(**inputs_data, max_new_tokens=512)
        inference_end_time = time.time()
        logging.info(f"Output generated in {inference_end_time - inference_start_time:.4f} seconds.")

        # 解码响应
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取响应部分
        response_start = "### Response:\n"
        final_response = response.split(response_start)[-1].strip()

        # 计算token数量
        num_input_tokens = len(tokenizer(input_text, return_tensors="pt").input_ids[0])
        num_output_tokens = len(tokenizer(final_response, return_tensors="pt").input_ids[0])

        # 清理 GPU 内存
        clear_cuda_cache()

        # 返回响应
        return jsonify({
            "generated_text": final_response,
            "num_output_tokens": num_output_tokens,
            "num_input_tokens": num_input_tokens
        })

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
