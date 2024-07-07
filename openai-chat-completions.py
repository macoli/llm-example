# 导入OpenAI库和dotenv库，用于初始化OpenAI客户端和加载环境变量
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 加载环境变量文件中的设置，通常包含OpenAI API密钥
load_dotenv(find_dotenv())

# 初始化OpenAI客户端
client = OpenAI()

# 指定使用的模型，这里是GPT-3.5 Turbo
model="openai/gpt-3.5-turbo"

# 定义聊天会话的消息列表，包含系统、助手和用户的消息
# system: 系统消息有助于设定助手的行为。在上面的例子中，助手被说明为“你是一个能干的助手”。
# user: 用户消息帮助指示助手。它们可以由应用的用户生成，也可以由开发者设置为指令。
# assistant: 助手消息用于存储之前的响应。用于帮助助手记住之前对话的信息（上下文）。
messages = [
    {"role": "system", "content": "你是一个能干的智能助手"},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "深圳今天天气怎么样？"}
]

# 使用OpenAI客户端的chat completions接口创建聊天回复
# 这里使用了指定的模型、消息列表、最大生成令牌数和温度参数
completion = client.chat.completions.create(
  model=model,
  messages=messages,
  max_tokens=512,
  temperature=0.7,
)

# 打印生成的聊天回复内容
print(completion.choices[0].message.content)

