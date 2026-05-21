import re
import math


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'
    
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'

def verify(stored_answer: str, predicted: str) -> bool:
    
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()