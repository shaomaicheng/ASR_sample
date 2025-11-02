import pyttsx3

# 初始化引擎
engine = pyttsx3.init()

# 设置参数（可选）
engine.setProperty('rate', 150)    # 语速
engine.setProperty('volume', 0.9)  # 音量

# 要转换的文本
text = "你好，这是一个人工智能音频"

# 保存为 wav 文件（pyttsx3 不直接支持保存，需用临时文件）
import os
engine.save_to_file(text, 'chinese_output.wav')
engine.runAndWait()

print("✅ 已生成 chinese_output.wav")

