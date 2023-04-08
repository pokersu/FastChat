import argparse
import datetime
import os
import time
import requests
import gradio as gr

from loguru import logger
from chatbot.conversation import (default_conversation, conv_templates, SeparatorStyle)
from chatbot.css import code_highlight_css
from chatbot.patch import Chatbot as grChatbot
from chatbot.common import server_error_msg
from gradio.components import *
from markdown2 import Markdown
import nest_asyncio
from pyngrok import ngrok

logger.add("server.log")

headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(".", f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    state = default_conversation.copy()
    return (state, gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))

def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5

def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, temperature, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        template_name = "v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    worker_addr = args.worker_url
    logger.info(f"worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    pload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) + 1:].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    logger.info((state, state.to_gradio_chatbot()) + (disable_btn,) * 5)
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except Exception as e:
        logger.error("occur error ", e)
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


notice_markdown = ("""
# <center>🤖️ ChatBot</center>
<br/>
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")


css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""


def build_demo():
    with gr.Blocks(title="FastChat", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        # Draw layout
        gr.Markdown(notice_markdown)
        chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=650)
        with gr.Row():
            with gr.Column(scale=10):
                textbox = gr.Textbox(show_label=False,
                    placeholder="Enter text and press ENTER", visible=False).style(container=False)
            with gr.Column(scale=1, min_width=60):
                submit_btn = gr.Button(value="Submit", visible=False)

        with gr.Row(visible=False) as button_row:
            regenerate_btn = gr.Button(value="Regenerate", interactive=False)
            clear_btn = gr.Button(value="Clear history", interactive=False)

        with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

        # gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        regenerate_btn.click(regenerate, state,
            [state, chatbot, textbox] + btn_list).then(
            http_bot, [state, temperature, max_output_tokens],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

        textbox.submit(add_text, [state, textbox], [state, chatbot, textbox] + btn_list
            ).then(http_bot, [state, temperature, max_output_tokens],
                   [state, chatbot] + btn_list)
        submit_btn.click(add_text, [state, textbox], [state, chatbot, textbox] + btn_list
            ).then(http_bot, [state, temperature, max_output_tokens],
                   [state, chatbot] + btn_list)

        demo.load(load_demo, [url_params], [state, chatbot, textbox, submit_btn, button_row, parameter_row],
                  _js=get_window_url_params)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--worker-url", type=str, default="http://127.0.0.1:21002")
    parser.add_argument("--concurrency-count", type=int, default=4)
    parser.add_argument("--moderate", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    demo = build_demo()
    #ngrok.set_auth_token("2NpamagUIzuzbAw3kqUnG1dmaha_5rHqwEGpAE4yYZ3osBLcK")
    #ngrok_tunnel = ngrok.connect(8080)
    #logger.info(f'Public URL: {ngrok_tunnel.public_url}')
    #nest_asyncio.apply()
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False)\
        .launch(
        server_name=args.host, server_port=args.port, share=False)

