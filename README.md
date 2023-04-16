# 开源LLM&数据集汇总

## 1. 开源LLM

### 1.1 LLaMA-7B
- 规模: 7B
- 申请链接: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- 接口: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)

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
## 3. 开放大模型

### 3.1 Claude
- 预备应用: Slack，Claude在slack中使用，需要首先注册[slack](https://www.slack.com)，建议使用gmail注册
- 应用申请链接: [https://anthropic.com/claude-in-slack](https://anthropic.com/claude-in-slack)，进入后，点击**add to slack**即可使用
- 接口申请链接: [https://www.anthropic.com/product](https://www.anthropic.com/product)，点击**request access**，填写表单申请
