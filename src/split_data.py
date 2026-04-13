import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

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

    print(f"Загрузка данных из {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if 'label' not in df.columns:
        raise ValueError("В датасете отсутствует колонка 'label' для проведения стратификации.")

    # Обработка редких классов (для stratify требуется минимум 2 примера на класс)
    class_counts = df['label'].value_counts()
    single_item_classes = class_counts[class_counts < 2].index
    if len(single_item_classes) > 0:
        print(f"\n[!] Внимание: Исключены классы с 1 примером (невозможно стратифицировать):")
        print(list(single_item_classes))
        df = df[~df['label'].isin(single_item_classes)]

    print(f"\nРазбиение данных (val_size = {args.val_size})...")
    train_df, val_df = train_test_split(
        df, 
        test_size=args.val_size, 
        random_state=args.seed, 
        stratify=df['label']
    )

    print(f"Сохранение train в {args.train_path} ({len(train_df)} строк)")
    train_df.to_csv(args.train_path, index=False)
    
    print(f"Сохранение val в {args.val_path} ({len(val_df)} строк)")
    val_df.to_csv(args.val_path, index=False)

    # Подсчет процентов
    train_dist = train_df['label'].value_counts(normalize=True) * 100
    val_dist = val_df['label'].value_counts(normalize=True) * 100

    # Объединение в один DataFrame для наглядности
    dist_df = pd.DataFrame({
        'Train %': train_dist,
        'Val %': val_dist
    }).fillna(0).round(2)

    # Сортировка по убыванию доли в тренировочной выборке
    dist_df.sort_values(by='Train %', ascending=False, inplace=True)

    print("\n" + "="*50)
    print("РАСПРЕДЕЛЕНИЕ ЗАДАЧ (LABEL) В ВЫБОРКАХ")
    print("="*50)
    print(dist_df.to_string())
    print("="*50)

if __name__ == "__main__":
    main()