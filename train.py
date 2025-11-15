import inspect
from dataset_train import load_dataset
import json
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          Seq2SeqTrainingArguments, 
                          DataCollatorForSeq2Seq, 
                          Seq2SeqTrainer)

from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

max_input_length = 128
max_target_length = 64

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], max_length=max_input_length, truncation=True
    )
    labels = tokenizer(
        examples["target_text"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_dataset()

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets.column_names, 
)

print(f"Number of tokenized examples: {len(tokenized_datasets)}")
print("\nSample tokenized dataset item:")
print(tokenized_datasets[0])

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    if hasattr(labels, 'tolist'): 
        labels = labels.tolist()
    decoded_labels = tokenizer.batch_decode([[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels], skip_special_tokens=True)

    json_valid_preds = []
    exact_matches = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        is_pred_json = False
        try:
            json.loads(pred)
            is_pred_json = True
        except json.JSONDecodeError:
            pass

        if is_pred_json:
            json_valid_preds.append(1)

        if pred == label:
            exact_matches += 1

    json_valid_pred_ratio = sum(json_valid_preds) / len(json_valid_preds) if json_valid_preds else 0
    exact_match_ratio = exact_matches / len(decoded_preds) if decoded_preds else 0

    return {
        "json_valid_pred_ratio": json_valid_pred_ratio,
        "exact_match_ratio": exact_match_ratio,
    }

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v", "k", "o", "wi_0","wi_1", "wo"],
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

print("LoRA configuration applied and trainable parameters displayed.")

class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): 
       
        sig = inspect.signature(model.forward)
        forward_args = set(sig.parameters.keys())


        filtered_inputs = {k: v for k, v in inputs.items() if k in forward_args}


        labels = None
        if self.label_smoother is not None and "labels" in filtered_inputs:
            labels = filtered_inputs.pop("labels")


        outputs = model(**filtered_inputs)


        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(filtered_inputs.keys())}."
            )
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs[0]

        if self.label_smoother is not None and labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)

        return (loss, outputs) if return_outputs else loss


train_test_split_ratio = 0.2
split_datasets = tokenized_datasets.train_test_split(test_size=train_test_split_ratio)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]


training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-small-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to=["wandb"], 
    run_name="flan-t5-extraction-lora",
    optim="adamw_torch", 
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = CustomSeq2SeqTrainer( 
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics, 
)

trainer.train()
