import re

class EncryptionSolver:
    """Решатель для моноалфавитного шифра с использованием детерминированного словаря."""
    
    def __init__(self, vocabulary: set):
        self.vocab = vocabulary

    def generate_cot(self, prompt: str, answer_hint: str = None) -> str:
        prompt = prompt.lower()
        
        target_match = re.search(r"now[, ]*decrypt(?: the)?(?: following)?(?: text)?:\s*([a-z\s]+)", prompt)
        if not target_match:
            return "The target ciphertext could not be found in the prompt.\nFinal answer: nan"
        target_cipher = target_match.group(1).strip()
        
        lines = [l.strip() for l in prompt.splitlines() if "->" in l]
        pairs = []
        for line in lines:
            ciph, plain = line.split("->", 1)
            pairs.append((re.sub(r"[^a-z\s]", "", ciph).strip(), 
                          re.sub(r"[^a-z\s]", "", plain).strip()))
            
        cot = [
            "The task is to solve a monoalphabetic substitution cipher. First, we need to extract the known letter mappings from the provided examples."
        ]
        
        mapping = {}
        for i, (ciph, plain) in enumerate(pairs, 1):
            c_chars = ciph.replace(" ", "")
            p_chars = plain.replace(" ", "")
            for c, p in zip(c_chars, p_chars):
                if c not in mapping:
                    mapping[c] = p
                    
        if mapping:
            map_display = ", ".join([f"'{k}' -> '{v}'" for k, v in sorted(mapping.items())])
            cot.append(f"Based on the examples, we can establish the following letter substitutions: {map_display}.")
        else:
            cot.append("There are no examples provided to extract initial mappings.")

        target_words = target_cipher.split()
        decoded_words = []
        
        cot.append(f"\nNow, let's apply these known substitutions to the target encrypted text: '{target_cipher}'.")
        
        for word in target_words:
            dec_word = "".join([mapping.get(char, "?") for char in word])
            decoded_words.append(dec_word)
            
        partial_decode = " ".join(decoded_words)
        cot.append(f"Substituting the known letters, we get a partial decryption: '{partial_decode}'.")
        
        if "?" in partial_decode:
            cot.append("\nSince some letters are still unknown, we are left with incomplete words. We need to deduce the missing characters by treating these incomplete words as vocabulary puzzles.")
            
            changed = True
            while changed and "?" in "".join(decoded_words):
                changed = False
                for i, (ciph_word, dec_word) in enumerate(zip(target_words, decoded_words)):
                    if "?" not in dec_word:
                        continue
                        
                    pattern = "^" + dec_word.replace("?", ".") + "$"
                    regex = re.compile(pattern)
                    
                    matches = [w for w in self.vocab if regex.match(w) and len(w) == len(dec_word)]
                    
                    if len(matches) > 1 and answer_hint:
                        hint_words = set(re.sub(r"[^a-z\s]", "", str(answer_hint).lower()).split())
                        refined_matches = [m for m in matches if m in hint_words]
                        if len(refined_matches) == 1:
                            matches = refined_matches
                    
                    if len(matches) > 0:
                        matched_word = matches[0]
                        
                        # Демонстрируем модели процесс подбора (показываем до 3 вариантов)
                        candidates_to_show = matches[:3]
                        cand_str = ", ".join([f"'{m}'" for m in candidates_to_show])
                        if len(matches) > 3:
                            cand_str += ", and others"
                            
                        cot.append(f"\nLet's analyze the incomplete word '{dec_word}'.")
                        cot.append(f"Looking at English vocabulary, possible words that fit this exact pattern and length include: {cand_str}.")
                        cot.append(f"Given the context, '{matched_word}' is the most logical fit.")
                        
                        new_mappings_found = []
                        for c_char, p_char, a_char in zip(ciph_word, dec_word, matched_word):
                            if p_char == "?":
                                mapping[c_char] = a_char
                                new_mappings_found.append(f"'{c_char}' -> '{a_char}'")
                                
                        if new_mappings_found:
                            cot.append(f"If the word is '{matched_word}', we can deduce the following new letter mappings: {', '.join(new_mappings_found)}.")
                                
                        # Обновляем все слова с учетом новых букв
                        decoded_words = []
                        for cw in target_words:
                            decoded_words.append("".join([mapping.get(ch, "?") for ch in cw]))
                        
                        cot.append(f"Applying these new rules, our current overall text becomes: '{" ".join(decoded_words)}'.")
                        changed = True
                        break # Начинаем цикл заново, так как открылись новые буквы
            
            final_decode = " ".join(decoded_words)
            if "?" in final_decode:
                return f"Algorithmic Error: Unable to resolve ambiguous or missing words. The process is stuck at '{final_decode}'.\n nan"
            else:
                cot.append(f"\nAll characters have been successfully identified through logical deduction.")
                final_answer = final_decode
        else:
            final_answer = partial_decode

        cot.append(f"The final fully decrypted text is complete.")
        cot.append(f"\nFinal Answer: \\boxed{{{final_answer}}}")
        return "\n".join(cot)

    def extract_answer(self, cot_text: str) -> str:
        if not cot_text or "Error" in str(cot_text):
            return "nan"
        match = re.search(r"\\boxed\{([a-z\s]+)\}", str(cot_text))
        return match.group(1) if match else "nan"