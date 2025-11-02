from huggingface_hub import snapshot_download
snapshot_download(
        repo_id="csukuangfj/streaming-paraformer-zh",
        local_dir="model",  # 下载到 ./model 目录
        allow_patterns=["*.onnx", "*.txt", "*.mvn", "*.yaml"]
    )