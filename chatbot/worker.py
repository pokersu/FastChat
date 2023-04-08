"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from chatbot.common import server_error_msg
import torch
import uvicorn

GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
logger.add("worker.log")
global_counter = 0

model_semaphore = None


def load_model(model_path, num_gpus):
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
       model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, **kwargs)

    model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    print(context_len)

    return tokenizer, model, context_len


class ModelWorker:
    def __init__(self, worker_addr, worker_id,
                 model_path, model_name, num_gpus):
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.context_len = load_model(model_path, num_gpus)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + len(
                model_semaphore._waiters)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        logger.info(f"max_token: {max_new_tokens}, input_ids len {len(input_ids)}")
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        del past_key_values

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status():
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--model-path", type=str, default="models/vicuna-13b")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(f"http://{args.host}:{args.port}",
                         worker_id,
                         args.model_path,
                         args.model_name,
                         args.num_gpus)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
