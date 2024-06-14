from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

app = Flask(__name__)

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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HyperdustProtocol/HyperAoto-llama2-7b-818",
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    input_text = data.get("input_text")

    if not input_text:
        return jsonify({"error": "Please provide both  input_text"}), 400

    inputs = tokenizer(
        [alpaca_prompt.format('Please generate a motion script according to the following description', input_text,
                              "")],
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
    app.run(host='0.0.0.0', port=5100)
