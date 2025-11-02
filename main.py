import numpy as np
import onnxruntime as ort
import librosa

# ----------------------------
# 1. 加载词表
# ----------------------------
def load_tokens(token_file):
    tokens = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token:  # 忽略空行
                tokens[idx] = token
    return tokens

# ----------------------------
# 2. 音频预处理（FBank + CMVN）
# ----------------------------
def extract_fbank(audio_path, n_mels=80, frame_length=25, frame_shift=10, sr=16000):
    # Load audio
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    # FBank
    fbank = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels,
        n_fft=int(sr * 0.001 * frame_length),
        hop_length=int(sr * 0.001 * frame_shift),
        fmin=20, fmax=8000
    )
    fbank = np.log(fbank + 1e-6).T  # (T, 80)
    return fbank.astype(np.float32)

def apply_cmvn(cmvn_file):
    # 读取 am.mvn（Kaldi 格式）
    with open(cmvn_file, "r") as f:
        lines = f.readlines()
    neg_mean = None
    inv_stddev = None
    for line in lines:
        if line.startswith("<LearnRateCoef>"):
            parts = line.strip().split()[3:-1]
            if neg_mean is None:
                neg_mean = np.array([float(x) for x in parts], dtype=np.float32)
            else:
                inv_stddev = np.array([float(x) for x in parts], dtype=np.float32)
    return neg_mean, inv_stddev

# ----------------------------
# 3. LFR 特征拼接（Lookahead +拼接）
# ----------------------------
def apply_lfr(features, lfr_m=5, lfr_n=1):
    # LFR: 每 lfr_n 帧取 lfr_m 帧拼接
    T, D = features.shape
    LFR_features = []
    for t in range(0, T, lfr_n):
        # 取 [t - (lfr_m-1)//2, t + (lfr_m-1)//2] 范围，不足补零
        start = max(0, t - (lfr_m - 1) // 2)
        end = min(T, t + (lfr_m - 1) // 2 + 1)
        chunk = features[start:end]
        # 补零到 lfr_m 行
        if chunk.shape[0] < lfr_m:
            pad = np.zeros((lfr_m - chunk.shape[0], D), dtype=np.float32)
            if t < (lfr_m - 1) // 2:  # 前面补
                chunk = np.concatenate([pad, chunk], axis=0)
            else:  # 后面补
                chunk = np.concatenate([chunk, pad], axis=0)
        LFR_features.append(chunk.flatten())
    return np.stack(LFR_features, axis=0)  # (T', 80*lfr_m)

# ----------------------------
# 4. 主推理函数
# ----------------------------
def main():
    audio_path = "chinese_output.wav"  # ← 替换为你的音频
    tokens = load_tokens("model/tokens.txt")
    vocab_size = len(tokens)

    # 特征提取
    # 1. 提取原始 fbank (T, 80)
    fbank = extract_fbank(audio_path)  # 不要在这里做 CMVN！

    # 2. 应用 LFR → (T', 560)
    feats = apply_lfr(fbank, lfr_m=7, lfr_n=6)  # 注意：用 7 和 6！

    # 3. 加载 560 维 CMVN 参数
    neg_mean, inv_stddev = apply_cmvn("model/am.mvn")

    # 4. 对 LFR 后的特征做 CMVN
    feats = (feats + neg_mean) * inv_stddev  # 现在 feats 是 (T', 560)，neg_mean 是 (560,)

    # 准备输入
    speech = feats[np.newaxis, :, :]  # (1, T, 400) → 注意：LFR后是 80*5=400
    speech_lengths = np.array([feats.shape[0]], dtype=np.int32)

    # 加载 encoder
    encoder_sess = ort.InferenceSession("model/model.onnx")
    enc_out = encoder_sess.run(
        ["enc", "enc_len", "alphas"],
        {"speech": speech, "speech_lengths": speech_lengths}
    )
    enc, enc_len, alphas = enc_out

    # 生成 acoustic embeddings（简化：用 enc 代替）
    acoustic_embeds = enc
    acoustic_embeds_len = enc_len

    # 初始化 decoder cache（共16层，每层 [1, 512, 10]）
    cache_dim = 10
    cache = [np.zeros((1, 512, cache_dim), dtype=np.float32) for _ in range(16)]

    # decoder 输入
    decoder_inputs = {
        "enc": enc,
        "enc_len": enc_len,
        "acoustic_embeds": acoustic_embeds,
        "acoustic_embeds_len": acoustic_embeds_len,
    }
    for i in range(16):
        decoder_inputs[f"in_cache_{i}"] = cache[i]

    # 加载 decoder
    decoder_sess = ort.InferenceSession("model/decoder.onnx")
    outputs = decoder_sess.run(None, decoder_inputs)

    logits = outputs[0]  # (1, T, 8404)
    sample_ids = outputs[1]  # (1, T)

    # 解码

    # print(logits)
    print(sample_ids)
    # for idx in sample_ids[0]:
    #     print(tokens.get(idx))
    text = "".join([tokens.get(int(idx), "<unk>") for idx in sample_ids[0]])
    print("识别结果:", text)


main()



# # 1. 加载音频
# y, sr = librosa.load("chinese_output.wav", sr=16000)
# print(f"✅ 音频: {len(y)/sr:.2f}秒, 采样率={sr}")

# # 2. 提取 FBank
# fbank = librosa.feature.melspectrogram(
#     y=y, sr=16000, n_mels=80,
#     n_fft=400, hop_length=160, fmin=20, fmax=8000
# )
# fbank = np.log(fbank + 1e-6).T.astype(np.float32)
# print("✅ FBank shape:", fbank.shape)

# # 3. LFR (m=7, n=6)
# T, D = fbank.shape
# lfr_m, lfr_n = 7, 6
# lfr_feats = []
# t = 0
# while t < T:
#     chunk = fbank[t:t+lfr_m]
#     if len(chunk) < lfr_m:
#         chunk = np.pad(chunk, ((0, lfr_m - len(chunk)), (0, 0)), mode='constant')
#     lfr_feats.append(chunk.flatten())
#     t += lfr_n
# lfr_feats = np.stack(lfr_feats) if lfr_feats else np.zeros((0, 560))
# print("✅ LFR shape:", lfr_feats.shape)

# # 4. 加载 CMVN
# with open("model/am.mvn") as f:
#     lines = f.readlines()
# neg_mean = inv_stddev = None
# for line in lines:
#     if line.startswith("<LearnRateCoef>"):
#         data = np.array([float(x) for x in line.split()[3:-1]], dtype=np.float32)
#         if neg_mean is None:
#             neg_mean = data
#         else:
#             inv_stddev = data
# print("✅ CMVN shape:", neg_mean.shape)

# # 5. 应用 CMVN
# feats = (lfr_feats + neg_mean) * inv_stddev
# print("✅ CMVN后 stats: mean=%.3f, std=%.3f" % (feats.mean(), feats.std()))

# # 6. 加载词表
# tokens = {}
# with open("model/tokens.txt", encoding="utf-8") as f:
#     for idx,line in enumerate(f):
#         token =line.strip()
#         if token:
#             tokens[idx] = token
# print("✅ 词表大小:", len(tokens))

# # 7. 加载模型并推理（简化）
# encoder = ort.InferenceSession("model/model.onnx")
# enc_out = encoder.run(None, {"speech": feats[None, :], "speech_lengths": np.array([len(feats)], dtype=np.int32)})
# enc = enc_out[0]  # (1, T, 512)
# print("✅ Encoder output shape:", enc.shape)