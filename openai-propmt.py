from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 加载环境变量文件中的设置，通常包含OpenAI API密钥
load_dotenv(find_dotenv())

openai = OpenAI()


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    instruction = """

    当向用户介绍流量套餐产品时，客服人员必须准确提及产品名称、月费价格和月流量总量

    已知产品包括：

    经济套餐：月费50元，月流量10G
    畅游套餐：月费180元，月流量100G
    无限套餐：月费300元，月流量1000G
    校园套餐：月费150元，月流量200G，限在校学生办理
    """

    # 输出描述
    output_format = """
    以JSON格式输出。
    """

    context = """
    你们有什么流量大的套餐
    """

    prompt = f"""
    {instruction}

    {output_format}

    对话记录：
    {context}
    """

    response = get_completion(prompt)
    print(response)
