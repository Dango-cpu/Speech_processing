import torch


def get_device():
    """
    Trả về thiết bị chạy mô hình.
    Ưu tiên CUDA nếu có.
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def get_torch_dtype():
    """
    Chọn dtype phù hợp theo thiết bị.
    """
    if torch.cuda.is_available():
        # GPU NVIDIA
        return torch.float16

    # CPU
    return torch.float32


def print_device_info():
    device = get_device()

    print("=" * 50)
    print(f"Device : {device}")

    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA   : {torch.version.cuda}")

    print("=" * 50)