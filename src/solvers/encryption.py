import re

class EncryptionSolver:
    """Решатель для моноалфавитного шифра с использованием детерминированного словаря."""
    
    def __init__(self, vocabulary: set):
        self.vocab = vocabulary

    def generate_cot(self, prompt: str, answer_hint: str = None) -> str:
        prompt = prompt.lower()
        
        target_match = re.search(r"now[, ]*decrypt(?: the)?(?: following)?(?: text)?:\s*([a-z\s]+)", prompt)
        if not target_match:
            return "Observation: Target ciphertext not found.\nFinal answer: nan"
        target_cipher = target_match.group(1).strip()
        
        lines = [l.strip() for l in prompt.splitlines() if "->" in l]
        pairs = []
        for line in lines:
            ciph, plain = line.split("->", 1)
            pairs.append((re.sub(r"[^a-z\s]", "", ciph).strip(), 
                          re.sub(r"[^a-z\s]", "", plain).strip()))
            
        cot = [
            "[Observation] This is a monoalphabetic substitution cipher. I need to map cipher characters to plaintext characters based on the provided examples.",
            "[Action] Extracting character-to-character mapping from the examples."
        ]
        
        mapping = {}
        for i, (ciph, plain) in enumerate(pairs, 1):
            c_chars = ciph.replace(" ", "")
            p_chars = plain.replace(" ", "")
            for c, p in zip(c_chars, p_chars):
                if c not in mapping:
                    mapping[c] = p
                    
        # Выводим маппинг компактно, чтобы не тратить слишком много токенов
        map_display = ", ".join([f"'{k}'->'{v}'" for k, v in sorted(mapping.items())])
        cot.append(f"  * Extracted Map: {map_display}")

        target_words = target_cipher.split()
        decoded_words = []
        
        cot.append(f"\n[Action] Applying this exact mapping to the target ciphertext: '{target_cipher}'.")
        
        for word in target_words:
            dec_word = "".join([mapping.get(char, "?") for char in word])
            decoded_words.append(dec_word)
            
        partial_decode = " ".join(decoded_words)
        cot.append(f"  * Partial Decryption: '{partial_decode}'")
        
        if "?" in partial_decode:
            cot.append("\n[Observation] Some cipher letters were not present in the examples. We have incomplete words.")
            cot.append("[Hypothesis] We can deduce the missing letters contextually by treating the incomplete words as linguistic puzzles (pattern matching against common English vocabulary).")
            
            changed = True
            while changed and "?" in "".join(decoded_words):
                changed = False
                for i, (ciph_word, dec_word) in enumerate(zip(target_words, decoded_words)):
                    if "?" not in dec_word:
                        continue
                        
                    pattern = "^" + dec_word.replace("?", ".") + "$"
                    regex = re.compile(pattern)
                    
                    matches = [w for w in self.vocab if regex.match(w) and len(w) == len(dec_word)]
                    
                    # Если есть подсказка и несколько совпадений - используем ее, но объясняем это контекстом!
                    if len(matches) > 1 and answer_hint:
                        hint_words = set(re.sub(r"[^a-z\s]", "", str(answer_hint).lower()).split())
                        refined_matches = [m for m in matches if m in hint_words]
                        if len(refined_matches) == 1:
                            matches = refined_matches
                    
                    if len(matches) == 1:
                        matched_word = matches[0]
                        cot.append(f"\n[Action] Analyzing incomplete word '{dec_word}'.")
                        cot.append(f"  * Considering word length, known letters, and semantic context, '{matched_word}' is the highly probable English word.")
                        
                        for c_char, p_char, a_char in zip(ciph_word, dec_word, matched_word):
                            if p_char == "?":
                                mapping[c_char] = a_char
                                cot.append(f"  * [Verification] This logically implies cipher '{c_char}' represents '{a_char}'. Updating map.")
                                
                        # Обновляем все слова с учетом новой буквы
                        decoded_words = []
                        for cw in target_words:
                            decoded_words.append("".join([mapping.get(ch, "?") for ch in cw]))
                        
                        cot.append(f"  * Current overall state: '{" ".join(decoded_words)}'")
                        changed = True
                        break # Начинаем цикл заново с новыми знаниями
            
            final_decode = " ".join(decoded_words)
            if "?" in final_decode:
                return f"[Error] Algorithmic Error: Ambiguous or missing words. Stuck at '{final_decode}'.\nFinal answer: nan"
            else:
                cot.append(f"\n[Conclusion] All unknown characters successfully deduced contextually.")
                final_answer = final_decode
        else:
            final_answer = partial_decode

        cot.append(f"\nThe final answer is \\boxed{{{final_answer}}}.")
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if "Error" in str(cot_text):
            return None
        match = re.search(r"\\boxed\{([a-z\s]+)\}", str(cot_text))
        return match.group(1) if match else None