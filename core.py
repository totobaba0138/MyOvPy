import torch
import open_clip
import sys

ML_CONTEXT = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None
}


class ModelConfig:
    # 显存够用 ViT-H，不够改 ViT-L-14
    ARCH = 'ViT-H-14'
    WEIGHTS = 'laion2b_s32b_b79k'


def load_global_model():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n🚀 [Core] Loading Model: {ModelConfig.ARCH} on {device}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            ModelConfig.ARCH, pretrained=ModelConfig.WEIGHTS, device=device
        )
        tokenizer = open_clip.get_tokenizer(ModelConfig.ARCH)

        ML_CONTEXT.update({
            "model": model, "preprocess": preprocess,
            "tokenizer": tokenizer, "device": device
        })
        print("✅ [Core] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [Core] Model Load Error: {e}")
        sys.exit(1)


def unload_global_model():
    ML_CONTEXT.clear()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("🛑 [Core] Model unloaded.")