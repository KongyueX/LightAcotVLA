import time
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)

# Go2ACOTInputs 支持 len(state) == 183 或 159
state = np.zeros((183,), dtype=np.float32)

# Go2ACOTInputs 需要 images 字典，且必须包含 top_head / hand_left / hand_right
# 图像格式用 uint8, HWC
dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

obs = {
    "state": state,
    "images": {
        "top_head": dummy_img,
        "hand_left": dummy_img,
        "hand_right": dummy_img,
    },
    "prompt": "Clean the desktop.",
}

for i in range(5):
    t0 = time.time()
    out = policy.infer(obs)
    dt = time.time() - t0

    print(f"\n[{i}] infer time: {dt:.3f}s")
    print("output type:", type(out))

    if isinstance(out, dict):
        print("output keys:", out.keys())
        for k, v in out.items():
            try:
                arr = np.asarray(v)
                print(f"{k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
            except Exception:
                print(f"{k}: {type(v)}")
    else:
        print(out)
