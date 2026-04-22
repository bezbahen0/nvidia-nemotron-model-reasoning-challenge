import re


class EncryptionSolver:    
    def __init__(self, vocabulary: set):
        self.vocab = vocabulary

    def generate_cot(self, prompt: str, answer_hint: str = None) -> str:
        prompt = prompt.lower()
        
        target_match = re.search(r"now[, ]*decrypt(?: the)?(?: following)?(?: text)?:\s*([a-z\s]+)", prompt)
        if not target_match:
            return "Parse Error: Target not found."
        target_cipher = target_match.group(1).strip()
        
        lines = [l.strip() for l in prompt.splitlines() if "->" in l]
        pairs = []
        for line in lines:
            ciph, plain = line.split("->", 1)
            pairs.append((re.sub(r"[^a-z\s]", "", ciph).strip(), 
                          re.sub(r"[^a-z\s]", "", plain).strip()))
            
        cot = ["Let's decrypt the text by building a letter mapping from the examples.\n"]
        mapping = {}
        
        for i, (ciph, plain) in enumerate(pairs, 1):
            c_chars = ciph.replace(" ", "")
            p_chars = plain.replace(" ", "")
            for c, p in zip(c_chars, p_chars):
                if c not in mapping:
                    mapping[c] = p
                    
        cot.append("Extracted mapping:")
        for k in sorted(mapping.keys()):
            cot.append(f"  {k} -> {mapping[k]}")

        target_words = target_cipher.split()
        decoded_words = []
        
        cot.append(f"\nNow translating target ciphertext: '{target_cipher}'")
        
        for word in target_words:
            dec_word = "".join([mapping.get(char, "?") for char in word])
            decoded_words.append(dec_word)
            
        partial_decode = " ".join(decoded_words)
        cot.append(f"Direct substitution gives: '{partial_decode}'")
        
        if "?" in partial_decode:
            cot.append("\nSome letters are missing. We must deduce them using standard English vocabulary and word patterns.")
            
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
                    
                    if len(matches) == 1:
                        matched_word = matches[0]
                        cot.append(f"  Looking at the incomplete word '{dec_word}', the only valid English word that fits this exact pattern in context is '{matched_word}'.")
                        
                        for c_char, p_char, a_char in zip(ciph_word, dec_word, matched_word):
                            if p_char == "?":
                                mapping[c_char] = a_char
                                cot.append(f"  Therefore, we can logically deduce that cipher '{c_char}' represents '{a_char}'.")
                                
                        decoded_words = []
                        for cw in target_words:
                            decoded_words.append("".join([mapping.get(ch, "?") for ch in cw]))
                        changed = True
                        break
            
            final_decode = " ".join(decoded_words)
            if "?" in final_decode:
                return f"Algorithmic Error: Ambiguous or missing words. Stuck at '{final_decode}'."
            else:
                cot.append(f"\nAll letters successfully deduced.")
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