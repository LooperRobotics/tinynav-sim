"""SigLIP2 image-tower TRT wrapper (L2-baked plan, unit-norm embeddings).

Preprocessing mirrors SigLIPImageTRT.preprocess_image (models_trt.py): BGR ->
RGB, INTER_CUBIC resize to 224, /255 then (x-0.5)/0.5. The deployed SigLIP1
and SigLIP2 share this contract, which the 0.2 benchmark validated.
"""
import numpy as np
import tensorrt as trt
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _check(*results):
    for r in results:
        if isinstance(r, tuple) and int(r[0]) != 0:
            raise RuntimeError(f"cudart error: {r}")


class Siglip2ImageEngine:
    def __init__(self, plan_path: str):
        with open(plan_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]
        self.in_name = None
        self.out_name = None
        self._out_host = None
        self._out_device = None
        self._in_host = None
        self._in_device = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        shape = self.engine.get_tensor_shape(self.in_name)
        self.net_h, self.net_w = int(shape[2]), int(shape[3])

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_CUBIC)
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return (x - 0.5) / 0.5

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(self.preprocess(bgr).reshape(-1))
        if self._in_host is None:
            self._in_host = np.empty(x.size, dtype=np.float32)
            self._in_device = cudart.cudaMalloc(self._in_host.nbytes)[1]
            self.context.set_tensor_address(self.in_name, self._in_device)
            out_shape = self.engine.get_tensor_shape(self.out_name)
            self._out_host = np.empty(int(np.prod(out_shape)), dtype=np.float32)
            self._out_device = cudart.cudaMalloc(self._out_host.nbytes)[1]
            self.context.set_tensor_address(self.out_name, self._out_device)
        np.copyto(self._in_host, x)
        _check(cudart.cudaMemcpyAsync(
            self._in_device, self._in_host.ctypes.data, self._in_host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("siglip2 image enqueue failed")
        _check(cudart.cudaMemcpyAsync(
            self._out_host, self._out_device, self._out_host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))
        emb = self._out_host.copy()
        n = float(np.linalg.norm(emb))
        if not 0.99 <= n <= 1.01:
            raise RuntimeError(f"siglip2 embedding norm {n:.4f} outside [0.99, 1.01]")
        return emb
