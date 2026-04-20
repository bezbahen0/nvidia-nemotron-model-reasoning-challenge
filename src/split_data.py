import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.log import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Скрипт для стратифицированного разбиения датасета")
    
    parser.add_argument("--data_path", type=str, required=True, help="Путь к исходному train.csv")
    parser.add_argument("--train_path", type=str, default="./data/train_split.csv", help="Путь для сохранения train")
    parser.add_argument("--val_path", type=str, default="./data/val_split.csv", help="Путь для сохранения val")
    
    parser.add_argument("--val_size", type=float, default=0.1, help="Доля данных для валидации (например, 0.1 для 10%)")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info(f"Load data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if 'label' not in df.columns:
        raise ValueError("Where is 'label' in CoT data?")

    class_counts = df['label'].value_counts()
    single_item_classes = class_counts[class_counts < 2].index
    if len(single_item_classes) > 0:
        logger.info(f"\n[!] Внимание: Исключены классы с 1 примером (невозможно стратифицировать):")
        logger.info(list(single_item_classes))
        df = df[~df['label'].isin(single_item_classes)]

    logger.info(f"val_size = {args.val_size}...")
    train_df, val_df = train_test_split(
        df, 
        test_size=args.val_size, 
        random_state=args.seed, 
        stratify=df['label']
    )

    logger.info(f"Save train в {args.train_path} ({len(train_df)} строк)")
    train_df.to_csv(args.train_path, index=False)
    
    logger.info(f"Save val в {args.val_path} ({len(val_df)} строк)")
    val_df.to_csv(args.val_path, index=False)

    train_dist = train_df['label'].value_counts(normalize=True) * 100
    val_dist = val_df['label'].value_counts(normalize=True) * 100

    dist_df = pd.DataFrame({
        'Train %': train_dist,
        'Val %': val_dist
    }).fillna(0).round(2)

    dist_df.sort_values(by='Train %', ascending=False, inplace=True)

    logger.info(f"\n{dist_df.to_string()}")
    
if __name__ == "__main__":
    main()