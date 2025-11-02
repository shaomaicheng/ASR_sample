import numpy as np
import onnx
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
    waveform, _ = librosa.load(audio_path, sr=sr)
    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    # FBank
    fbank = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels,
        n_fft=int(sr * 0.001 * frame_length),
        hop_length=int(sr * 0.001 * frame_shift),
        fmin=20, fmax=8000
    )
    # print(fbank.shape)
    fbank = np.log(fbank + 1e-6).T  # (T, 80)
    # print(fbank.shape)
    return fbank.astype(np.float32)

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
        LFR_features.append(chunk.flatten()) # 多个形状相同的数组“打包”成一个更高维的数组。
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

    model = onnx.load("new-model/model.onnx")
    meta = {prop.key: prop.value for prop in model.metadata_props}

    # 2. 应用 LFR → (T', 560)
    lfr_m = int(meta["lfr_window_size"])
    lfr_n = int(meta["lfr_window_shift"])
    # print("lfr_m:", lfr_m)
    # print("lfr_n:", lfr_n)
    feats = apply_lfr(fbank, lfr_m, lfr_n)  # 注意：用 7 和 6！


    # 3. 加载 560 维 CMVN 参数
    neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
    inv_stddev = np.array([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)

    # 4. 对 LFR 后的特征做 CMVN归一化
    feats = feats * inv_stddev + neg_mean # 现在 feats 是 (T', 560)，neg_mean 是 (560,)

    # 准备输入
    speech = feats[np.newaxis, :, :]  # (1, T, 400) → 注意：LFR后是 80*5=400

    speech_lengths = np.array([feats.shape[0]], dtype=np.int32)

    # 加载 encoder
    encoder_sess = ort.InferenceSession("new-model/model.onnx")
    enc_out = encoder_sess.run(
        ["enc", "enc_len", "alphas"],
        {"speech": speech, "speech_lengths": speech_lengths}
    )
    enc, enc_len, alphas = enc_out
    # 加载 decoder
    decoder_sess = ort.InferenceSession("model/decoder.onnx")
    # cache
    cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
    decoder_inputs = {
        "enc": enc,
        "enc_len": enc_len,
        "acoustic_embeds": enc,
        "acoustic_embeds_len": enc_len,
    }
    for i in range(16):
        decoder_inputs[f"in_cache_{i}"] = cache[i]
    outputs = decoder_sess.run(None, decoder_inputs)
    logits = outputs[0]
    sample_ids = outputs[1]
    text = "".join([tokens.get(int(idx), "<unk>") for idx in sample_ids[0]])
    print(text)

    # 循环
    # chunk_size = 100
    # all_token_ids=[]
    # T = enc.shape[1]
    # offset = 0
    # in_cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
    # # 3. 流式解码
    # while offset < T:
    #     end = min(offset + chunk_size, T)
    #     enc_chunk = enc[:, offset:end, :]
    #     L = end - offset
    #     enc_len_np = np.array([L], dtype=np.int32)
    #
    #     inputs = {
    #         "enc": enc_chunk,
    #         "enc_len": enc_len_np,
    #         "acoustic_embeds": enc_chunk,
    #         "acoustic_embeds_len": enc_len_np,
    #     }
    #     for i in range(16):
    #         inputs[f"in_cache_{i}"] = in_cache[i]
    #
    #     outputs = decoder_sess.run(None, inputs)
    #     in_cache = [out.copy() for out in outputs[2:18]]
    #     offset = end
    #     clean_ids = []
    #     for tid in outputs[1][0]:
    #         if not clean_ids or tid != clean_ids[-1]:
    #             clean_ids.append(tid)
    #     for i in clean_ids:
    #         all_token_ids.append(i)
    # # 转文本
    # text = "".join(tokens.get(tid, "<unk>") for tid in all_token_ids)
    # print(text)

main()