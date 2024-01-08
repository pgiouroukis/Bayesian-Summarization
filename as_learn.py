import argparse
import json
import logging
import os
import sys
import random
import time

import torch

from src.active_summarization import active_sum
from src.common.loaders import init_dataset
from src.common.logging_utils import set_global_logging_level

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],)

set_global_logging_level(logging.ERROR, ["transformers.configuration_utils"])
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--train_file", type=str, help="")
    parser.add_argument("--validation_file", type=str, help="")
    parser.add_argument("--dataset_name", type=str, help="")
    parser.add_argument("--dataset_config_name", type=str, help="")
    parser.add_argument("--text_column", type=str, help="")
    parser.add_argument("--summary_column", type=str, help="")
    parser.add_argument("--seed", type=int, default=10, help="")
    parser.add_argument("--resume", type=int, default=0, help="")

    # AS args
    parser.add_argument("--N", type=int, default=10, help="")
    parser.add_argument("--L", type=int, default=20, help="")
    parser.add_argument("--K", type=int, default=100, help="")
    parser.add_argument("--S", type=int, default=10, help="")
    parser.add_argument("--steps", type=int, default=10, help="")
    parser.add_argument("--acquisition", type=str, default="bayesian", choices=["bayesian", "random"], help="")
    parser.add_argument("--preacquisition", type=str, choices=["idds"], help="")
    parser.add_argument("--preacquisition_samples", type=int, help="")
    parser.add_argument("--embeddings_model", type=str, help="")

    # Training args
    parser.add_argument("--init_model", type=str, help="")
    parser.add_argument("--max_source_length", type=int, default=256, help="")
    parser.add_argument("--max_summary_length", type=int, default=62, help="")
    parser.add_argument("--training_validation", type=int, default=1, help="")
    parser.add_argument("--max_val_samples", type=int, help="")
    parser.add_argument("--max_test_samples", type=int, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--batch_size_eval", type=int, default=8, help="")
    parser.add_argument("--num_beams", type=int, default=3, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--save_step", type=int, default=100, help="")
    parser.add_argument("--save_limit", type=int, default=1, help="")
    parser.add_argument("--metric_for_best_model", type=str, default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                        help="")

    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    # CUDA or MPS for PyTorch
    st = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'    
    torch.backends.cudnn.benchmark = True

    args, unknown = read_args()

    args.resume = bool(args.resume)
    args.training_validation = bool(args.training_validation)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    data_path = os.path.join(args.output_path, "data")
    if args.resume:
        logger.info("Resuming session")
        if not os.path.exists(data_path):
            logger.info("Data dir not found. Unable to resume session.")
            sys.exit()
    else:
        if os.path.exists(data_path):
            logger.info("Data dir already exists. Aborting.")
            sys.exit()
        os.mkdir(data_path)

    train_model = os.path.join(args.output_path, "models")

    train_dataset = init_dataset(
        data_path=args.train_file,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split="train")
    train_sampler = active_sum.DataSampler(train_dataset, split="train")
    
    if not args.resume:
        validation_dataset = init_dataset(
            data_path=args.validation_file,
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split="validation")
        validation_sampler = active_sum.DataSampler(validation_dataset, split="validation")
        validation_samples, validation_samples_idx = validation_sampler.sample_data(args.max_val_samples)
        validation_sampler.remove_samples(validation_samples_idx)
        outf = open(os.path.join(args.output_path+"/data", "validation.json"), "a+")
        for di, (data_s, data_idx, si) in enumerate(zip(validation_samples, validation_samples_idx, validation_samples_idx)):
            validation_sample_json = {
                "document": data_s[args.text_column].replace('\n', ' '),
                "summary": data_s[args.summary_column].replace('\n', ' '),
                "id": data_idx
            }
            json.dump(validation_sample_json, outf)
            outf.write("\n")
        outf.close()

    logger.info(f"{args.acquisition} L: {args.L} K: {args.K} S: {args.S} steps: {args.steps}")

    if args.acquisition == "bayesian":
        active_learner = active_sum.BAS(
            train_sampler,
            device=device,
            doc_col=args.text_column,
            sum_col=args.summary_column,
            seed=args.seed,
            py_module=__name__,
            init_model=args.init_model,
            source_len=args.max_source_length,
            target_len=args.max_summary_length,
            training_validation=args.training_validation,
            val_samples=args.max_val_samples,
            test_samples=args.max_test_samples,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,            
            beams=args.num_beams,
            lr=args.learning_rate,
            save_step=args.save_step,
            save_limit=args.save_limit,
            metric=args.metric_for_best_model,
            embeddings_model=args.embeddings_model,
            preacquisition=args.preacquisition,
            preacquisition_samples=args.preacquisition_samples,
        )
    else:
        active_learner = active_sum.RandomActiveSum(
            train_sampler,
            device=device,
            doc_col=args.text_column,
            sum_col=args.summary_column,
            seed=args.seed,
            py_module=__name__,
            init_model=args.init_model,
            source_len=args.max_source_length,
            target_len=args.max_summary_length,
            training_validation=args.training_validation,
            val_samples=args.max_val_samples,
            test_samples=args.max_test_samples,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            beams=args.num_beams,
            lr=args.learning_rate,
            save_step=args.save_step,
            save_limit=args.save_limit,
            metric=args.metric_for_best_model,
        )

    if args.resume:
        active_learner.resume_learner(data_path)
    else:
        logger.info("#FIND: initializing bayesian learning")
        active_learner.init_learner(
            init_labeled=args.L,
            model_path=train_model,
            labeled_path=data_path,
            eval_path=args.validation_file,
            epochs=args.epochs)

    if args.acquisition == "bayesian":
        logger.info("#FIND: running bayesian learning")
        active_learner.learn(
            steps=args.steps,
            model_path=train_model,
            labeled_path=data_path,
            k=args.K, s=args.S, n=args.N,
            eval_path=args.validation_file,
            epochs=args.epochs)
    else:
        active_learner.learn(
            steps=args.steps,
            model_path=train_model,
            labeled_path=data_path,
            k=args.K, s=args.S,
            eval_path=args.validation_file,
            epochs=args.epochs)

    et = time.time()
    logger.info(f"Elapsed time: {et - st} sec.")


if __name__ == "__main__":
    main()
