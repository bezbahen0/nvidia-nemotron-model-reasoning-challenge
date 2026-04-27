import random
import string
import pandas as pd
import re

from src.solvers.encryption import EncryptionSolver

encryption_prompt_template = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
{examples}
Now, decrypt the following text: {target_cipher}"""


class EncryptionTaskGenerator:
    """Генератор синтетических задач для моноалфавитного шифра."""
    
    def __init__(self, vocabulary: set, seed: int = None):
        self.vocab = list(vocabulary)
        self.encryption_solver = EncryptionSolver(vocabulary)
        self.rng = random.Random(seed)
        self.alphabet = list(string.ascii_lowercase)

    def _generate_cipher_map(self):
        """Создает случайный словарь подстановок (a-z -> a-z)."""
        shuffled = self.alphabet.copy()
        self.rng.shuffle(shuffled)
        # Возвращаем два словаря: plain->cipher и cipher->plain
        return dict(zip(self.alphabet, shuffled)), dict(zip(shuffled, self.alphabet))

    def _encrypt(self, text: str, enc_map: dict) -> str:
        """Шифрует текст по заданной карте подстановок."""
        result = []
        for char in text.lower():
            if char in enc_map:
                result.append(enc_map[char])
            elif char.isspace():
                result.append(char)
        return "".join(result)

    def generate_task(self, num_examples=5, words_per_sentence=(3, 6)):
        """
        Генерирует задачу.
        :param num_examples: Количество примеров перед таргетом
        :param words_per_sentence: Диапазон количества слов в предложении
        :return: (formatted_examples_str, target_plaintext, target_ciphertext)
        """
        enc_map, dec_map = self._generate_cipher_map()

        # Вспомогательная функция для генерации предложений из словаря
        def gen_sentence():
            count = self.rng.randint(*words_per_sentence)
            return " ".join(self.rng.choices(self.vocab, k=count))

        examples = []
        for _ in range(num_examples):
            plain = gen_sentence()
            cipher = self._encrypt(plain, enc_map)
            examples.append((cipher, plain))

        target_plain = gen_sentence()
        target_cipher = self._encrypt(target_plain, enc_map)

        examples_str = "\n".join([f"{c} -> {p}" for c, p in examples])
        
        return examples_str, target_plain, target_cipher

    def generate_dataset(self, num_samples: int):
        """Создает датасет заданного размера в формате DataFrame."""
        results = []
        for i in range(num_samples):
            examples_line, answer, target_cipher = self.generate_task(
                num_examples=self.rng.randint(2, 6) 
            )
            
            prompt = encryption_prompt_template.format(
                examples=examples_line, 
                target_cipher=target_cipher
            )
            
            generated_cot = self.encryption_solver.generate_cot(prompt)
            computed_answer = self.encryption_solver.extract_answer(generated_cot)

            results.append({
                "prompt": prompt,
                "answer": answer,
                "generated_cot": generated_cot,
                "computed_answer": computed_answer,
                "label": "encryption"
            })
            
        return pd.DataFrame.from_records(results)


if __name__ == "__main__":

    sample_vocab = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
        "alice", "secret", "message", "crypto", "hello", "world", "solve"
    }

    generator = EncryptionTaskGenerator(vocabulary=sample_vocab, seed=42)
    
    results = []
    for i in range(3):
        examples_line, answer, target_cipher = generator.generate_task(num_examples=4)
        prompt = encryption_prompt_template.format(
            examples=examples_line, 
            target_cipher=target_cipher
        )
        
        cot = generator.encryption_solver.generate_cot(prompt)
        computed_answer = generator.encryption_solver.extract_answer(cot)
        
        print(f"--- Task {i+1} ---")
        print(prompt)
        print(f"\nExpected: {answer}")
        print(f"Computed: {computed_answer}")
        print("-" * 40)

        results.append(1 if answer == computed_answer else 0)
    
    print(f"\nAccuracy: {sum(results) / float(len(results)):.2f}")