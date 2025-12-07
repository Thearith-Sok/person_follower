import torch
import argparse
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config

def export_to_onnx(model_path, label_path, onnx_path):

    # ------------------------------
    # 1. Load labels
    # ------------------------------
    class_names = [l.strip() for l in open(label_path)]
    num_classes = len(class_names)
    print("📌 Number of classes:", num_classes)

    # ------------------------------
    # 2. Build SSD model
    # ------------------------------
    print("📦 Creating MobileNet-SSD network...")
    net = create_mobilenetv1_ssd(num_classes=num_classes, is_test=True)

    print("📦 Loading weights from:", model_path)
    state_dict = torch.load(model_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()

    # ------------------------------
    # 3. Prepare dummy input (300x300)
    # ------------------------------
    dummy = torch.randn(1, 3,
                        mobilenetv1_ssd_config.image_size,
                        mobilenetv1_ssd_config.image_size)

    # ------------------------------
    # 4. Export ONNX
    # ------------------------------
    print("🚀 Exporting ONNX →", onnx_path)
    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["scores", "boxes"],
        opset_version=11,
        do_constant_folding=True,
    )

    print("✅ ONNX export complete:", onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("label_path")
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.label_path, args.onnx)