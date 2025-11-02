from typing import Dict

import numpy as np
import onnx

import yaml


def load_cmvn():
    neg_mean = None
    inv_stddev = None

    with open("model/am.mvn") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def get_vocab_size():
    with open("model/tokens.txt") as f:
        return len(f.readlines())


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.
    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    new_filename = f"new-{filename}"
    onnx.save(model, new_filename)
    print(f"Updated {new_filename}")


def main():
    with open("model/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    lfr_window_size = config["frontend_conf"]["lfr_m"]
    lfr_window_shift = config["frontend_conf"]["lfr_n"]
    encoder_output_size = config["encoder_conf"]["output_size"]
    decoder_num_blocks = config["decoder_conf"]["num_blocks"]
    decoder_kernel_size = config["decoder_conf"]["kernel_size"]
    cif_threshold = config["predictor_conf"]["threshold"]
    tail_threshold = config["predictor_conf"]["tail_threshold"]

    neg_mean, inv_stddev = load_cmvn()
    vocab_size = get_vocab_size()

    meta_data = {
        "lfr_window_size": str(lfr_window_size),
        "lfr_window_shift": str(lfr_window_shift),
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "encoder_output_size": encoder_output_size,
        "decoder_num_blocks": decoder_num_blocks,
        "decoder_kernel_size": decoder_kernel_size,
        "model_type": "paraformer",
        "version": "1",
        "model_author": "damo",
        "maintainer": "k2-fsa",
        "vocab_size": str(vocab_size),
        "comment": "speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
    }
    add_meta_data("model/model.onnx", meta_data)
    add_meta_data("model/model_quant.onnx", meta_data)


if __name__ == "__main__":
    main()