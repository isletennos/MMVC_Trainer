import os
import argparse
import time
import json

import onnx
from onnxsim import simplify
import onnxruntime as ort
import torch

from models import SynthesizerTrn
from text.symbols import symbols


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(
        checkpoint_path), f"No such file or directory: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model, optimizer, learning_rate, iteration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--convert_pth", required=True)
    return parser.parse_args()


def inspect_onnx(session):
    print("inputs")
    for i in session.get_inputs():
        print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))
    print("outputs")
    for i in session.get_outputs():
        print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))


def benchmark(session):
    dummy_specs = torch.rand(1, 257, 60)
    dummy_lengths = torch.LongTensor([60])
    dummy_sid_src = torch.LongTensor([0])
    dummy_sid_tgt = torch.LongTensor([1])

    use_time_list = []
    for i in range(30):
        start = time.time()
        output = session.run(
            ["audio"],
            {
                "specs": dummy_specs.numpy(),
                "lengths": dummy_lengths.numpy(),
                "sid_src": dummy_sid_src.numpy(),
                "sid_tgt": dummy_sid_tgt.numpy()
            }
        )
        use_time = time.time() - start
        use_time_list.append(use_time)
        #print("use time:{}".format(use_time))
    use_time_list = use_time_list[5:]
    mean_use_time = sum(use_time_list) / len(use_time_list)
    print(f"mean_use_time:{mean_use_time}")


class OnnxSynthesizerTrn(SynthesizerTrn):
    def forward(self, y, y_lengths, sid_src, sid_tgt):
        return self.voice_conversion(y, y_lengths, sid_src, sid_tgt)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat


def main(args):
    hps = get_hparams_from_file(args.config_file)

    net_g = OnnxSynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    for i in net_g.parameters():
        i.requires_grad = False
    _ = net_g.eval()
    _ = load_checkpoint(args.convert_pth, net_g, None)
    print("Model data loading succeeded.\nConverting start.")

    # Convert to ONNX
    dirname = os.path.dirname(args.convert_pth)
    filenames = os.path.splitext(os.path.basename(args.convert_pth))
    onnx_file = os.path.join(dirname, filenames[0] + ".onnx")
    dummy_specs = torch.rand(1, 257, 60)
    dummy_lengths = torch.LongTensor([60])
    dummy_sid_src = torch.LongTensor([0])
    dummy_sid_tgt = torch.LongTensor([1])
    torch.onnx.export(
        net_g,
        (dummy_specs, dummy_lengths, dummy_sid_src, dummy_sid_tgt),
        onnx_file,
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=["specs", "lengths", "sid_src", "sid_tgt"],
        output_names=["audio"],
        dynamic_axes={
            "specs": {2: "length"}
        })
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    print("Done\n")

    print("vits onnx benchmark")
    ort_session_cpu = ort.InferenceSession(
        onnx_file,
        providers=["CPUExecutionProvider"])
    ort_session_cuda = ort.InferenceSession(
        onnx_file,
        providers=["CUDAExecutionProvider"])
    inspect_onnx(ort_session_cpu)
    print("ONNX CPU")
    benchmark(ort_session_cpu)
    print("ONNX CUDA")
    benchmark(ort_session_cuda)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
