import re
import argparse
import pandas as pd

from src.augmentation.upsample import upsample_dataframe
from src.augmentation.bit_manipulation import BitManipulationTaskGenerator
from src.augmentation.encryption import EncryptionTaskGenerator
from src.metric import verify
from src.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    #parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    

    
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Load data from {args.data_path}...")
    data = pd.read_csv(args.data_path)



    # Remove all nan
    data = data[~data.computed_answer.isna()]

    # skip all answer that not verified
    data = data[data.apply(lambda row: verify(row["computed_answer"], row["answer"]), axis=1)]

    logger.info(f"Datset countes: \n{data.label.value_counts()}")

    multipliers = {
        "bit manipulation": 3.0,
        "": 2.0,
    }

    # Equations
    data = upsample_dataframe(data, multipliers_map={"equations transformation": 4.0}, seed=args.seed)

    # bit manipulation
    bit_manipulation_generator = BitManipulationTaskGenerator(seed=args.seed)

    bit_mp_gen_dataset = bit_manipulation_generator.generate_dataset(
        int(len(data[data.label == "bit manipulation"]) * multipliers["bit manipulation"])
    )

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
    print(global_vocab)

    encryption_gen_dataset = encryption_generator.generate_dataset(
        int(len(data[data.label == "encryption"]) * 3.0) # или любое нужное количество
    )


    data = pd.concat([encryption_gen_dataset, bit_mp_gen_dataset, data])

    data = data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    

    data = data[data.apply(lambda row: verify(row["computed_answer"], row["answer"]), axis=1)]
    logger.info(f"Final dataset len: \n{data.label.value_counts()}")

    data.to_csv(args.output_path, index=False)

    data["computed"] = data.apply(lambda row: verify(row["computed_answer"], row["answer"]), axis=1)

    logger.info(f'Complted accuracy: \n{data.groupby("label").computed.value_counts()}')

    
if __name__ == "__main__":
    main()