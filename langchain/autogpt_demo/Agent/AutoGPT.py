from typing import List, Optional, Tuple

from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import ValidationError

from Agent.Action import Action
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PrintUtils import *


def _format_long_term_memory(task_description: str, memory: BaseChatMemory) -> str:
    """
    格式化长时记忆。

    :param task_description: 任务描述。
    :param memory: 长时记忆对象。
    :return: 格式化后的长时记忆历史。
    """
    return memory.load_memory_variables(
        {"prompt": task_description}
    )["history"]


def _format_short_term_memory(memory: BaseChatMemory) -> str:
    """
    格式化短时记忆。

    :param memory: 短时记忆对象。
    :return: 格式化后的短时记忆消息。
    """
    messages = memory.chat_memory.messages
    string_messages = [messages[i].content for i in range(1, len(messages))]
    return "\n".join(string_messages)


class AutoGPT:
    """
    AutoGPT 类实现基于Langchain的对话流程控制。

    :param llm: 基础对话模型。
    :param prompts_path: 提示文件路径。
    :param tools: 工具列表。
    :param work_dir: 工作目录，默认为"./data"。
    :param main_prompt_file: 主提示文件名，默认为"main.json"。
    :param final_prompt_file: 最终提示文件名，默认为"final_step.json"。
    :param max_thought_steps: 最大思考步骤数，默认为10。
    :param memory_retriever: 向量存储检索器，默认为None。
    """
    def __init__(
            self,
            llm: BaseChatModel,
            prompts_path: str,
            tools: List[BaseTool],
            work_dir: str = "./data",
            main_prompt_file: str = "main.json",
            final_prompt_file: str = "final_step.json",
            max_thought_steps: Optional[int] = 10,
            memory_retriever: Optional[VectorStoreRetriever] = None,
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps
        self.memory_retriever = memory_retriever

        # OutputFixingParser： 如果输出格式不正确，尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)

        self.main_prompt_file = main_prompt_file
        self.final_prompt_file = final_prompt_file

    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        查找工具。

        :param tool_name: 工具名称。
        :return: 符合名称的工具对象，如果没有找到则返回None。
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _step(self,
              reason_chain,
              task_description,
              short_term_memory,
              long_term_memory,
              verbose=False
              ) -> Tuple[Action, str]:
        """
        执行一个思考步骤。

        :param reason_chain: 理由链。
        :param task_description: 任务描述。
        :param short_term_memory: 短时记忆。
        :param long_term_memory: 长时记忆。
        :param verbose: 是否详细输出，默认为False。
        :return: 行动和响应的元组。
        """
        """执行一步思考"""
        response = ""
        for s in reason_chain.stream({
            "short_term_memory": _format_short_term_memory(short_term_memory),
            "long_term_memory": _format_long_term_memory(task_description, long_term_memory)
            if long_term_memory is not None else "",
        }):
            if verbose:
                color_print(s, THOUGHT_COLOR, end="")
            response += s

        action = self.robust_parser.parse(response)
        return action, response

    def _final_step(self, short_term_memory, task_description) -> str:
        """
        执行最后一步，生成最终输出。

        :param short_term_memory: 短时记忆。
        :param task_description: 任务描述。
        :return: 最终输出。
        """
        """最后一步, 生成最终的输出"""
        finish_prompt = PromptTemplateBuilder(
            self.prompts_path,
            self.final_prompt_file,
        ).build().partial(
            task_description=task_description,
            short_term_memory=_format_short_term_memory(short_term_memory),
        )
        chain = (finish_prompt | self.llm | StrOutputParser())
        response = chain.invoke({})
        return response

    def _exec_action(self, action: Action) -> str:
        """
        执行行动。

        :param action: 需要执行的行动。
        :return: 行动的观察结果。
        """
        # 查找工具
        tool = self._find_tool(action.name)
        # action_expr = format_action(action)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )

        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    def run(self, task_description, verbose=False) -> str:
        """
        运行对话流程。

        :param task_description: 任务描述。
        :param verbose: 是否详细输出，默认为False。
        :return: 对话的最终回复。
        """
        thought_step_count = 0  # 思考步数

        # 初始化模板
        prompt_template = PromptTemplateBuilder(
            self.prompts_path,
            self.main_prompt_file,
        ).build(
            tools=self.tools,
            output_parser=self.output_parser,
        ).partial(
            work_dir=self.work_dir,
            task_description=task_description,
        )

        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
        )

        short_term_memory.save_context(
            {"input": "\n初始化"},
            {"output": "\n开始"}
        )

        # 初始化LLM链
        chain = (prompt_template | self.llm | StrOutputParser())

        # 如果有长时记忆，加载长时记忆
        if self.memory_retriever is not None:
            long_term_memory = VectorStoreRetrieverMemory(
                retriever=self.memory_retriever,
            )
        else:
            long_term_memory = None

        reply = ""

        while thought_step_count < self.max_thought_steps:
            if verbose:
                color_print(f">>>>Round: {thought_step_count}<<<<", ROUND_COLOR)

            action, response = self._step(
                chain,
                task_description=task_description,
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
                verbose=verbose,
            )

            if action.name == "FINISH":
                if verbose:
                    color_print(f"\n----\nFINISH", OBSERVATION_COLOR)

                reply = self._final_step(short_term_memory, task_description)
                break

            observation = self._exec_action(action)

            if verbose:
                color_print(f"\n----\n结果:\n{observation}", OBSERVATION_COLOR)

            # 保存到短时记忆
            short_term_memory.save_context(
                {"input": response},
                {"output": "返回结果:\n" + observation}
            )

            thought_step_count += 1

        if not reply:
            reply = "抱歉，我没能完成您的任务。"

        if long_term_memory is not None:
            # 保存到长时记忆
            long_term_memory.save_context(
                {"input": task_description},
                {"output": reply}
            )

        return reply
