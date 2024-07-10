# 导入必要的模块和类，用于构建提示模板和处理文件操作
import json
import os
import tempfile
from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import load_prompt
from langchain.schema.output_parser import BaseOutputParser
from langchain.tools.base import BaseTool
from langchain.tools.render import render_text_description
# 导入核心库的模板和工具类，用于支持提示模板的构建和处理
from langchain_core.prompts import PipelinePromptTemplate, BasePromptTemplate
from langchain_core.tools import Tool

# 导入动作类，用于示例中的输出解析器配置
from Agent.Action import Action


def _chinese_friendly(string) -> str:
    """
    将字符串中的JSON片段转换为中文友好的格式。

    参数:
    string (str): 包含JSON片段的字符串。

    返回:
    str: 转换后的字符串，其中的JSON片段已转义为中文。
    """
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)


def _load_file(filename: str) -> str:
    """
    加载并返回文件的内容。

    参数:
    filename (str): 文件的路径。

    返回:
    str: 文件的内容。

    异常:
    FileNotFoundError: 如果文件不存在，则抛出此异常。
    """
    """Loads a file into a string."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    f = open(filename, 'r', encoding='utf-8')
    s = f.read()
    f.close()
    return s


class PromptTemplateBuilder:
    """
    提示模板构建器类，用于根据配置文件和相关变量构建提示模板。

    参数:
    prompt_path (str): 提示模板文件所在的目录路径。
    prompt_file (str): 主提示模板文件的名称。
    """
    def __init__(
            self,
            prompt_path: str,
            prompt_file: str,
    ):
        self.prompt_path = prompt_path
        self.prompt_file = prompt_file

    def _check_or_redirect(self, prompt_file: str) -> str:
        """
        检查提示模板文件的配置，并根据需要重定向到临时文件。

        参数:
        prompt_file (str): 提示模板文件的路径。

        返回:
        str: 经过检查或重定向后的文件路径。

        说明:
        如果模板文件中配置了template_path，且为相对路径，则将其转换为绝对路径，
        并生成一个临时文件包含这个配置，返回这个临时文件的路径。
        """
        with open(prompt_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "template_path" in config:
            # 如果是相对路径，则转换为绝对路径
            if not os.path.isabs(config["template_path"]):
                config["template_path"] = os.path.join(self.prompt_path, config["template_path"])
                # 生成临时文件
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix='.json',
                    mode="w",
                    encoding="utf-8",
                    delete=False
                )
                tmp_file.write(json.dumps(config, ensure_ascii=False))
                tmp_file.close()
                return tmp_file.name
        return prompt_file

    def build(
            self,
            tools: Optional[List[BaseTool]] = None,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        """
        构建提示模板。

        参数:
        tools (Optional[List[BaseTool]]): 可用的工具列表，默认为None。
        output_parser (Optional[BaseOutputParser]): 输出解析器，默认为None。

        返回:
        BasePromptTemplate: 构建后的提示模板。

        说明:
        此方法会根据提供的工具和输出解析器来填充模板中的变量，并处理任何嵌套的模板。
        """
        # 加载主提示模板文件
        main_file = os.path.join(self.prompt_path, self.prompt_file)
        main_prompt_template = load_prompt(
            self._check_or_redirect(main_file)
        )
        variables = main_prompt_template.input_variables
        partial_variables = {}
        recursive_templates = []

        # 遍历所有变量，检查是否存在对应的模板文件
        for var in variables:
            # 处理嵌套的提示模板
            if os.path.exists(os.path.join(self.prompt_path, f"{var}.json")):
                sub_template = PromptTemplateBuilder(
                    self.prompt_path, f"{var}.json"
                ).build(tools=tools, output_parser=output_parser)
                recursive_templates.append((var, sub_template))
            # 处理纯文本变量
            elif os.path.exists(os.path.join(self.prompt_path, f"{var}.txt")):
                var_str = _load_file(os.path.join(self.prompt_path, f"{var}.txt"))
                partial_variables[var] = var_str

        # 如果提供了工具列表，填充tools变量
        if tools is not None and "tools" in variables:
            tools_prompt = render_text_description(tools)  # _get_tools_prompt(tools)
            partial_variables["tools"] = tools_prompt

        # 如果提供了输出解析器，填充format_instructions变量
        if output_parser is not None and "format_instructions" in variables:
            partial_variables["format_instructions"] = _chinese_friendly(
                output_parser.get_format_instructions()
            )

        # 处理嵌套模板的组合
        if recursive_templates:
            # 将有值嵌套的模板填充到主模板中
            main_prompt_template = PipelinePromptTemplate(
                final_prompt=main_prompt_template,
                pipeline_prompts=recursive_templates
            )

        # 填充部分变量并返回最终的提示模板
        # 将有值的变量填充到模板中
        main_prompt_template = main_prompt_template.partial(**partial_variables)

        return main_prompt_template


if __name__ == "__main__":
    # 示例：构建提示模板并进行格式化输出
    builder = PromptTemplateBuilder("../prompts/main", "main.json")
    output_parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = builder.build(tools=[
        Tool(name="FINISH", func=lambda: None, description="任务完成")
    ], output_parser=output_parser)
    print(prompt_template.format(
        task_description="解决问题",
        work_dir=".",
        short_term_memory="",
        long_term_memory="",
    ))
