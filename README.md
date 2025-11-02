# 桌面
python，使用模型：<font style="color:#080808;background-color:#ffffff;">csukuangfj/streaming-paraformer-zh</font>

这个是通义的模型导出的oxxn格式。

[https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)


第一步：使用onnx查看模型的输入输出：

![](https://)

![](https://)

```plain
model.onnx输入
speech: shape=['batch_size', 'feats_length', 560], type=tensor(float)
speech_lengths: shape=['batch_size'], type=tensor(int32)
```

表示输入参数speech是一个张量，张量大小是 shape ['batch_size', 'feats_length', 560] 指的是 batch_size个 feats_length行 560列的矩阵

batch_size指的批处理数量，demo里就是1

feats_length值得是特征值长度，例如每段语音提取多少帧

560：特征维度的数量（什么是特征维度，描述一个事物的特征用了多少个数字）



第二步，处理音频，获取向量输入

1. 特征值处理

使用librosa

加载音频文件、计算<font style="color:#080808;background-color:#ffffff;">梅尔频谱图得到一个矩阵</font>

<font style="color:#080808;background-color:#ffffff;">把梅尔频谱的结果做对数运算和矩阵转置</font>

2. 使用低帧率特征处理方法处理第一步的数据

旨在减少输入序列的长度，同时尽量保留信息。通过将相邻的多个帧拼接在一起，可以有效减少模型输入的长度，从而加速训练和推理过程

3. 加载模型的<font style="color:#080808;background-color:#ffffff;">neg_mean和inv_stddev参数</font>
4. <font style="color:#080808;background-color:#ffffff;">CMVN归一化</font>
5. <font style="color:#080808;background-color:#ffffff;">增加维度，因为模型输入需要batch数，所以给归一化后的矩阵扩维度</font>
6. <font style="color:#080808;background-color:#ffffff;">构造encoder的输入参数，执行encoder，获取encoder输出</font>
7. <font style="color:#080808;background-color:#ffffff;">加载decoder，构造decoder的输入。decoder的输入会包含一部分encoder以及kvcache，cache第一次全部填充为0</font>
8. <font style="color:#080808;background-color:#ffffff;">获取decoder的输出，给到的tokenids去token词表里查询文字</font>

