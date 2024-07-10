import sys

from colorama import Fore, Style

# 定义不同类型的文本颜色常量
THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.BLUE
CODE_COLOR = Fore.WHITE


def color_print(text, color=None, end="\n"):
    """
    输出带颜色的文本。

    参数:
    text: 要输出的文本。
    color: 文本的颜色，使用colorama库中的颜色常量。
    end: 表示行尾的字符，默认为换行符。

    功能:
    根据传入的颜色参数，为文本添加颜色效果并输出。
    """
    # 根据是否指定了颜色，构造带颜色效果或不带颜色效果的文本
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    # 直接向标准输出写入文本内容，并确保内容立即刷新到输出设备
    sys.stdout.write(content)
    sys.stdout.flush()
