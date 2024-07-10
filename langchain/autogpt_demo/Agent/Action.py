from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    行动模型定义了执行某个操作的名称和参数。

    Attributes:
        name (str): 操作的名称，必填项，用于标识操作。
        args (Optional[Dict[str, Any]]): 操作的参数，可选项，以键值对形式存储，键为参数名，值为参数值。
    """
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")
