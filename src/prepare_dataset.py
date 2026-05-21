import re
import argparse
import hashlib
import pandas as pd

from transformers import AutoTokenizer

from src.augmentation.equations.cryptarithm_generator import CryptarithmTaskGenerator
from src.augmentation.equations.ast_brute_force_generator import ASTBruteForceTaskGenerator
from src.augmentation.bit_manipulation import BitManipulationTaskGenerator
from src.augmentation.encryption import EncryptionTaskGenerator
from src.metric import verify
from src.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    return parser.parse_args()

def make_generated_id(row: pd.Series) -> str:
    raw = "\n".join([
        str(row["source"]),
        str(row["label"]),
        str(row["prompt"]),
        str(row["answer"]),
        str(row["computed_answer"]),
        str(row["generated_cot"]),
    ])

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=12,
    ).hexdigest()

    return f"gen_{digest}"


def main():
    args = parse_args()

    logger.info(f"Load data from {args.data_path}...")
    data = pd.read_csv(args.data_path)

    # Remove all nan
    data = data[~data.computed_answer.isna()]
    data["source"] = len(data) * ["solver"]

    # skip all answer that not verified
    data = data[data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)]

    logger.info(f"Datset countes: \n{data.label.value_counts()}")

    multipliers = {
        "bit manipulation": 0.5,
        "cryptarithm": 0.7,
        "equations_numeric": 0.7,
        "encryption": 0.5,
    }

    # Equations
    equations_cryptarithm_generator = CryptarithmTaskGenerator(seed=args.seed)

    equations_cryptarithm_dataset = equations_cryptarithm_generator.generate_dataset(
        num_samples=int(len(data[data.label == "equations transformation"]) * multipliers["cryptarithm"]),
        mode="random",
        nb_workers=24,
        progress_bar=False,
    )
    logger.info("\nCryptarithm generator:")
    logger.info(equations_cryptarithm_dataset.columns.tolist())
    logger.info(equations_cryptarithm_dataset.task_mode.value_counts(normalize=True))
    logger.info(f'Accuracy: {equations_cryptarithm_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean()}')

    equations_ast_generator = ASTBruteForceTaskGenerator(seed=args.seed)

    equations_ast_dataset = equations_ast_generator.generate_dataset(
        num_samples=int(len(data[data.label == "equations transformation"]) * multipliers["equations_numeric"]),
        nb_workers=24,
        progress_bar=False,
        label="equations transformation"
    )
    logger.info("\nNumeric equations generator:")
    logger.info(equations_ast_dataset.columns.tolist())
    logger.info(equations_ast_dataset.task_mode.value_counts(normalize=True))
    logger.info(f'Accuracy: {equations_ast_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean()}')
    
    # bit manipulation
    bit_manipulation_generator = BitManipulationTaskGenerator(seed=args.seed)

    bit_mp_gen_dataset = bit_manipulation_generator.generate_dataset(
        int(len(data[data.label == "bit manipulation"]) * multipliers["bit manipulation"])
    )
    logger.info("\Bit manipulation generator:")
    logger.info(bit_mp_gen_dataset.columns.tolist())
    logger.info(bit_mp_gen_dataset.task_mode.value_counts(normalize=True))
    logger.info(f'Accuracy: {bit_mp_gen_dataset.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1).mean()}')

    # Encryption

    global_vocab = set()
    for prompt in data[data.label == "encryption"]['prompt']:
        lines = [l.strip() for l in prompt.lower().splitlines() if "->" in l]
        for line in lines:
            plain = line.split("->", 1)[1]
            words = re.sub(r"[^a-z\s]", "", plain).split()
            global_vocab.update(words)

    for ans in data[data.label == "encryption"]['answer']:
        if isinstance(ans, str):
             global_vocab.update(re.sub(r"[^a-z\s]", "", ans.lower()).split())

    encryption_generator = EncryptionTaskGenerator(vocabulary=global_vocab, seed=args.seed)

    encryption_gen_dataset = encryption_generator.generate_dataset(
        int(len(data[data.label == "encryption"]) *  multipliers["encryption"])
    )


    data_gen = pd.concat([equations_cryptarithm_dataset, equations_ast_dataset, encryption_gen_dataset, bit_mp_gen_dataset])
    data_gen = data_gen[["prompt","answer","label","generated_cot","computed_answer"]]
    data_gen["source"] = len(data_gen) * ["generated"]
    data_gen["id"] = data_gen.apply(make_generated_id, axis=1)

    data = pd.concat([data_gen, data])
     
    data = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    

    data = data[data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)]
    logger.info(f"Final dataset len: \n{data.label.value_counts()}")

    data.to_csv(args.output_path, index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    data["token_len"] = data.generated_cot.apply(tokenizer.encode).apply(len)
    logger.info(f"\nDataset token len:\n{data.groupby('label').token_len.describe()}")

    data["computed"] = data.apply(lambda row: verify(row["answer"], row["computed_answer"]), axis=1)

    logger.info(f'Compilted accuracy: \n{data.groupby("label").computed.value_counts()}')

    
if __name__ == "__main__":
    main()