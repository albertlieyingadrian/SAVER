from trl import SFTConfig, SFTTrainer
from data import saver_collate_fn

def build_sft_config(train_config):
    """
    Construct an SFTConfig object from the training configuration dictionary.
    """
    return SFTConfig(
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        warmup_ratio=train_config["warmup_ratio"],
        num_train_epochs=train_config["num_train_epochs"],
        learning_rate=train_config["learning_rate"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        logging_steps=train_config["logging_steps"],
        optim=train_config["optim"],
        weight_decay=train_config["weight_decay"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        seed=train_config["seed"],
        save_strategy=train_config["save_strategy"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        dataset_text_field=train_config["dataset_text_field"],
        dataset_kwargs=train_config["dataset_kwargs"],
        remove_unused_columns=train_config["remove_unused_columns"],
        output_dir=train_config["output_dir"],
    )


def build_trainer(spmodel, processor, train_dataset, train_config):
    """
    Build an SFTTrainer with the SAVER model, processed dataset, and SFT config.
    """
    # Convert config → SFTConfig
    sft_args = build_sft_config(train_config)

    # Create trainer for supervised fine-tuning
    return SFTTrainer(
        model=spmodel,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        args=sft_args,
        data_collator=lambda examples: saver_collate_fn(examples, processor),
    )