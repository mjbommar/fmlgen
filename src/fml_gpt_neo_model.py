# (C) Michael Bommarito, 2013-2022
# SPDX: Apache-2.0
# Data collected from fmylife.com ~2013-2014
import lzma

# imports
import os

# set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hf imports
import numpy
import torch
from datasets import Dataset
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)

# setup default parameters
default_model_version = "EleutherAI/gpt-neo-1.3B"
default_max_length = 40
default_batch_size = 4


def get_model(model_version: str = default_model_version, device: str = "cuda:0"):
    """
    Get a model
    :param model_version: The model version to use.
    :param device: The device to use.
    :return: The model.
    """
    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_version).to(device)
    model.config.pad_token_id = model.config.eos_token_id

    # return the model
    return model


def get_tokenizer(model_version: str = default_model_version):
    """
    Get a tokenizer
    :param model_version: The model version to use.
    :return: The tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_version)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_sample(
    file_path: str = "data/sample.txt.xz", num_samples: int = None
) -> list[bytes]:
    """
    Load a sample file
    :param file_path: The path to the sample XZ newline-delimited file.
    :param num_samples: The number of samples to load.
    :return: The list of samples.
    """
    # open lzma file and readlines
    with lzma.open(file_path, "rt", encoding="utf-8", errors="ignore") as sample_file:
        sample_list = sample_file.readlines()

    # return random sample if requested
    if num_samples is not None:
        sample_list = numpy.random.choice(sample_list, num_samples, replace=False)

    # return the sample list
    return sample_list


# fine tune the model
def process_sample(
    sample: str, tokenizer: GPT2Tokenizer, max_length: int = default_max_length
):
    """
    Process the given sample.

    :param sample: The sample to process.
    :param tokenizer: The tokenizer to use.
    :param max_length: The maximum length of the sample.
    :return: The processed sample.
    """
    # process the sample
    return tokenizer(
        sample, truncation=True, padding="max_length", max_length=max_length
    )


def get_dataset(
    samples: list[str], tokenizer: GPT2Tokenizer = None, keep_text: bool = False
) -> Dataset:
    """
    Get a dataset.
    :param samples: The samples to use.
    :param tokenizer: The tokenizer to use.
    :param keep_text: Whether to keep the text in the dataset.
    :return: The dataset.
    """
    # create a dataset from the samples
    input_id_list = []
    attn_mask_list = []
    for sample in samples:
        # process the sample
        processed_sample = process_sample(sample, tokenizer)

        # add the sample to the lists
        input_id_list.append(processed_sample["input_ids"])
        attn_mask_list.append(processed_sample["attention_mask"])

    # create the dataset
    if keep_text:
        sample_dataset = Dataset.from_dict(
            {
                "input_ids": input_id_list,
                "labels": input_id_list,
                "attention_mask": attn_mask_list,
                "idx": list(range(len(samples))),
                "text": samples,
            }
        )
    else:
        sample_dataset = Dataset.from_dict(
            {
                "input_ids": input_id_list,
                "labels": input_id_list,
                "attention_mask": attn_mask_list,
            }
        )

    # return the dataset
    return sample_dataset


def train_model(
    dataset: torch.Tensor,
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    epochs: int = 1,
    batch_size: int = default_batch_size,
    weight_decay: float = 2e-4,
    learning_rate: float = 3e-4,
):
    """
    Train a model
    :param dataset: The dataset to train on.
    :param model: The model to train.
    :param tokenizer: The tokenizer to use.
    :param epochs: The number of epochs to train for.
    :param batch_size: The batch size to use.
    :param weight_decay: The weight decay to use.
    :param learning_rate: The learning rate to use.
    :return: The trained model.
    """
    # create the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        warmup_steps=500,
        logging_steps=10,
        logging_strategy="steps",
        eval_steps=10,
        evaluation_strategy="steps",
        save_steps=1000,
        # have some fun experimenting or optimizing here
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    # create the trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # train-eval in-sample
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    # train the model
    trainer.train()

    # save the model
    trainer.save_model("./model")


def load_model(model_path: str = "./model"):
    """
    Load a model
    :param model_path:
    :return:
    """
    return GPTNeoForCausalLM.from_pretrained(model_path).to("cuda:0")


def generate_fml(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str = None,
    max_length: int = 100,
    top_k: int = 100,
    top_p: float = 0.9,
    temperature: float = None,
    num_samples: int = 1,
    device: str = "cuda:0",
):
    """
    Generate an FML
    :param model: The model to use.
    :param tokenizer: The tokenizer to use.
    :param prompt: The prompt to use.
    :param max_length: The maximum length of the FML.
    :param top_p: top_p value to use for sampling
    :param top_k: top_k value to use for sampling
    :param temperature: temp value to use for sampling
    :param num_samples: The number of samples to generate.
    :param device: The device to use.
    :return:
    """
    # tokenize the prompt
    if prompt is None:
        prompt_list = ["Today,", "Yesterday,"]
        prompt = numpy.random.choice(prompt_list)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device)

    # generate the FML
    # note: you can change the sampling strategy/parameters here
    output = model.generate(
        **tokenized_prompt,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=num_samples,
    )

    # decode the FML with token padding
    fml = tokenizer.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # return the FML
    return fml
