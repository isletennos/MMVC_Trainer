import argparse
import time
import onnxruntime as ort
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_vits_onnx", required=True)
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


def main(args):
    ort_session_cpu = ort.InferenceSession(
        args.input_vits_onnx,
        providers=["CPUExecutionProvider"])

    ort_session_cuda = ort.InferenceSession(
        args.input_vits_onnx,
        providers=["CUDAExecutionProvider"])

#    ort_session_tensorrt = ort.InferenceSession(
#        args.input_vits_onnx,
#        providers=["TensorrtExecutionProvider"])

    print("vits onnx benchmark")
    inspect_onnx(ort_session_cpu)
    print("ONNX CPU")
    benchmark(ort_session_cpu)
    print("ONNX CUDA")
    benchmark(ort_session_cuda)
#    print("ONNX TensorRT")
#    benchmark(ort_session_tensorrt)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
