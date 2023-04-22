# 开源LLM&数据集汇总

## 1. 开源LLM

### 1.1 LLaMA

- 规模: 7B, 13B, 30B, 65B
- 申请链接: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- 接口: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)

### 1.2 Vicuna

- 规模: 7B, 13B
- 项目地址: [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)
- 调用:
  先安装必要的包

  ```shell
  pip install fschat
  pip install pip3 install git+https://github.com/huggingface/transformers
  ```

  运行
  ```shell
  python -m fastchat.serve.cli --model-path /datas/huggingface/vicuna-7b --gpu-nums 8     # 7B版本
  python -m fastchat.serve.cli --model-path /datas/huggingface/vicuna-13b --gpu-nums 8    # 13B版本
  ```
- 代码调用:

  ```python
  """
  Chat with a model with command line interface.
  Usage:
  python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
  """
  import argparse
  import re

  from prompt_toolkit import PromptSession
  from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
  from prompt_toolkit.completion import WordCompleter
  from prompt_toolkit.history import InMemoryHistory
  from rich.console import Console
  from rich.markdown import Markdown
  from rich.live import Live

  from fastchat.serve.inference import chat_loop, ChatIO


  class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream, skip_echo_len: int):
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)


  class RichChatIO(ChatIO):
    def __init__(self):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(words=['!exit', '!reset'], pattern=re.compile('$'))
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None)
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                accumulated_text = outputs[skip_echo_len:]
                if not accumulated_text:
                    continue
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in accumulated_text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return outputs[skip_echo_len:]


  def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO()
    elif args.style == "rich":
        chatio = RichChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(args.model_path, args.device, args.num_gpus, args.max_gpu_memory,
            args.load_8bit, args.conv_template, args.temperature, args.max_new_tokens,
            chatio, args.debug)
    except KeyboardInterrupt:
        print("exit...")


  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
        help="The path to the weights")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max-gpu-memory", type=str, default="13GiB")
    parser.add_argument("--load-8bit", action="store_true",
        help="Use 8-bit quantization.")
    parser.add_argument("--conv-template", type=str, default=None,
        help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--style", type=str, default="simple",
                        choices=["simple", "rich"], help="Display style.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
  ```

### 1.3 Koala

- 官网: [https://bair.berkeley.edu/blog/2023/04/03/koala/](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- 权重地址: [https://drive.google.com/drive/folders/10f7wrlAFoPIy-TECHsx9DKIvbQYunCfl?usp=sharing](https://drive.google.com/drive/folders/10f7wrlAFoPIy-TECHsx9DKIvbQYunCfl?usp=sharing)

### 1.4 Flan-T5

+ 规模: small(80M), base(250M), large(780M), xl(3B), xxl(11B)
+ 论文：[https://arxiv.org/pdf/2210.11416.pdf](https://arxiv.org/pdf/2210.11416.pdf)
+ GitHub:  [https://github.com/google-research/t5x](https://github.com/google-research/t5x)
+ HuggingFace:  [https://huggingface.co/docs/transformers/model_doc/t5](https://huggingface.co/docs/transformers/model_doc/t5)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_card = "/datas/huggingface/Flan-T5/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_card)
tokenizer = AutoTokenizer.from_pretrained(model_card)

inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## 2. 数据集

### 2.1 ShareGPT

- 官网: [https://sharegpt.com/](https://sharegpt.com/)
- 项目链接: [https://github.com/domeccleston/sharegpt](https://github.com/domeccleston/sharegpt)
- 接口:
  - 添加新记录
    ```python
    const res = await fetch("https://sharegpt.com/api/conversations", {
      body: JSON.stringify(conversationData),
      headers: {
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    const { id } = await res.json();
    const url = `https://shareg.pt/${id}`; // short link to the ShareGPT post
    ```
  - 获取数据
    ```python
    await fetch(
      "https://sharegpt.com/api/conversations?type=new&page=2&search=python"
    );
    interface ConversationMeta {
      id: string; // unique id for the conversation
      title: string; // title of the conversation (first user prompt)
      avatar: string; // base64 encoded URI of the user's avatar
      saves: number; // number of times the conversation is saved on ShareGPT
      comments: number; // number of comments the conversation has on ShareGPT
      views: number; // number of times the conversation has been viewed on ShareGPT
      createdAt: Date; // timestamp when the conversation was creataed
    }
    [];
    ```

### 2.2 ShareGPT-zh

- 项目地址: [https://paratranz.cn/projects/6725](https://paratranz.cn/projects/6725)

### 2.3 HC3

- 文献: [https://arxiv.org/abs/2301.07597](https://arxiv.org/abs/2301.07597)
- 项目地址: [https://github.com/Hello-SimpleAI/chatgpt-comparison-detection](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)

## 3. 开放大模型

### 3.1 Claude

- 预备应用: Slack，Claude在slack中使用，需要首先注册[slack](https://www.slack.com)，建议使用gmail注册
- 应用申请链接: [https://anthropic.com/claude-in-slack](https://anthropic.com/claude-in-slack)，进入后，点击**add to slack**即可使用
- 接口申请链接: [https://www.anthropic.com/product](https://www.anthropic.com/product)，点击**request access**，填写表单申请
