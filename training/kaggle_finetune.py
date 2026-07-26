# ============================================================
# Aether Orchestrator Fine-Tuning v2 - FIXED
# ============================================================
# Changes from v1:
# - Uses Instruct model (has chat template)
# - 500 examples instead of 100
# - 200 training steps instead of 60
# - Dataset format fixed (no markdown fences)
# ============================================================

# --- Install dependencies ---
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps xformers trl peft accelerate bitsandbytes

# --- Training ---
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch, os, shutil

# 1. Load Model
print("Loading Qwen2.5-Coder-1.5B-Instruct in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Inject LoRA adapters
print("Injecting LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,               # Doubled from 16 for stronger adaptation
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Load dataset
dataset_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".jsonl"):
            dataset_path = os.path.join(root, f)
            break

if dataset_path is None:
    raise FileNotFoundError("Could not find .jsonl file!")

print(f"Found dataset at: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_chat_template(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat_template, batched=True)
print(f"Dataset ready: {len(dataset)} examples")

# 4. Train with MORE steps and higher LoRA rank
print("Starting training (200 steps)...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=30,
        num_train_epochs=3,         # Switched to epochs to cover the massive 1500 dataset
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Cosine decay for better convergence
        seed=3407,
        output_dir="outputs",
    ),
)

stats = trainer.train()
print(f"\nTraining complete! Final loss: {stats.training_loss:.4f}")

# 5. Export to GGUF
print("\nExporting to GGUF format...")
model.save_pretrained_gguf(
    "Aether-Orchestrator-1.5B",
    tokenizer,
    quantization_method="q4_k_m"
)

# 6. Copy GGUF to /kaggle/working for download
for folder in ["Aether-Orchestrator-1.5B", "Aether-Orchestrator-1.5B_gguf"]:
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".gguf") or f == "Modelfile":
                src = os.path.join(folder, f)
                dst = os.path.join("/kaggle/working", f)
                shutil.copy2(src, dst)
                size_mb = os.path.getsize(dst) / (1024*1024)
                print(f"Copied: {f} ({size_mb:.1f} MB)")

print(f"\n{'='*50}")
print(f"SUCCESS! Download files from the Output tab.")
print(f"{'='*50}")
