import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

model_path = "./models/llama-2-7b"  # à adapter selon ton dossier local

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)

# ✅ Préparation LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ✅ Chargement dataset
data_path = "data/combined_dataset.csv"
dataset = load_dataset("csv", data_files={"train": data_path})

# ✅ Pré-traitement
def preprocess_function(examples):
    inputs = ["Question: " + q for q in examples["Question"]]
    targets = ["Answer: " + a for a in examples["Answer"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# ✅ Arguments d’entraînement
training_args = TrainingArguments(
    output_dir="./results_llama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
)

# ✅ Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ✅ Sauvegarde
model.save_pretrained("./lora_llama_model")
tokenizer.save_pretrained("./lora_llama_model")
