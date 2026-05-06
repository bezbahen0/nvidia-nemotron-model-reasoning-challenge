import re
from typing import Optional

class PseudoMathStringSolver:
    """
    Солвер для задач типа Pseudo-Math (String/Format Logic).
    Ищет закономерности строковых манипуляций (конкатенация, перемежение,
    сдвиги ASCII, пересечения), игнорируя математические свойства чисел.
    """
    def __init__(self):
        # Используем \w вместо \d на случай, если строковые операции 
        # применяются к буквенным операндам
        self._numeric_re = re.compile(r"^(-?\w+)\s*([^\w\s]+)\s*(-?\w+)$")

    def solve(self, examples_text: str, target_text: str) -> Optional[str]:
        # 1. Парсинг искомого выражения
        qm = self._numeric_re.fullmatch(target_text.strip())
        if not qm: 
            return None
        q_a, q_op, q_b = qm.group(1), qm.group(2).strip(), qm.group(3)

        # 2. Парсинг и фильтрация примеров строго по целевому оператору
        parsed_exs = []
        for line in examples_text.split('\n'):
            if '=' not in line:
                continue
            lhs, rhs = line.split('=', 1)
            m = self._numeric_re.fullmatch(lhs.strip())
            if m:
                a, op, b = m.group(1), m.group(2).strip(), m.group(3)
                if op == q_op:
                    # Разрешаем буквы и цифры в ответе (важно для ascii_shift)
                    clean_rhs = "".join(c for c in rhs.strip() if c.isalnum() or c == '-')
                    if clean_rhs:
                        parsed_exs.append((a, b, clean_rhs))

        if not parsed_exs:
            return None

        # Вспомогательная функция для проверки сдвига ASCII
        def get_ascii_shift(a_str: str, out_str: str) -> Optional[int]:
            if len(a_str) != len(out_str) or len(a_str) == 0: 
                return None
            diffs = set(ord(o) - ord(i) for o, i in zip(out_str, a_str))
            return diffs.pop() if len(diffs) == 1 else None

        # 3. Перебор гипотез строковых преобразований
        best_pattern = None
        global_shift_val = None
        
        patterns = [
            'fwd_concat', 'rev_concat', 'rev_both', 'interleave', 
            'subtraction', 'intersection', 'first_last', 'overlap', 'ascii_shift'
        ]

        for pattern in patterns:
            all_match = True
            shift_val = None
            
            for a, b, expected_out in parsed_exs:
                match = False
                
                if pattern == 'fwd_concat' and expected_out == a + b: 
                    match = True
                elif pattern == 'rev_concat' and expected_out == b + a: 
                    match = True
                elif pattern == 'rev_both' and expected_out == a[::-1] + b[::-1]: 
                    match = True
                elif pattern == 'interleave':
                    # Улучшенное перемежение (поддерживает разную длину строк)
                    interleaved = "".join(i+j for i,j in zip(a,b)) + a[len(b):] + b[len(a):]
                    if expected_out == interleaved and interleaved != "": 
                        match = True
                elif pattern == 'subtraction':
                    res = "".join([c for c in a if c not in b])
                    if expected_out == res and expected_out != "": 
                        match = True
                elif pattern == 'intersection':
                    res = "".join([c for c in a if c in b])
                    if expected_out == res and expected_out != "": 
                        match = True
                elif pattern == 'first_last':
                    if len(a) > 0 and len(b) > 0 and expected_out == a[0] + b[-1]: 
                        match = True
                elif pattern == 'overlap':
                    if len(a) > 0 and len(b) > 0 and a[-1] == b[0]:
                        if expected_out == a + b[1:]: 
                            match = True
                    elif expected_out == a + b: 
                        match = True
                elif pattern == 'ascii_shift':
                    shift = get_ascii_shift(a, expected_out)
                    if shift is not None:
                        if shift_val is None: 
                            shift_val = shift
                        if shift_val == shift: 
                            match = True
                
                if not match:
                    all_match = False
                    break
            
            if all_match:
                best_pattern = pattern
                global_shift_val = shift_val
                break

        # 4. Применение найденного правила к целевому запросу
        if best_pattern:
            if best_pattern == 'fwd_concat': return q_a + q_b
            elif best_pattern == 'rev_concat': return q_b + q_a
            elif best_pattern == 'rev_both': return q_a[::-1] + q_b[::-1]
            elif best_pattern == 'interleave': 
                return "".join(i+j for i,j in zip(q_a, q_b)) + q_a[len(q_b):] + q_b[len(q_a):]
            elif best_pattern == 'subtraction': return "".join([c for c in q_a if c not in q_b])
            elif best_pattern == 'intersection': return "".join([c for c in q_a if c in q_b])
            elif best_pattern == 'first_last': return q_a[0] + q_b[-1] if (len(q_a)>0 and len(q_b)>0) else ""
            elif best_pattern == 'overlap': return q_a + q_b[1:] if (len(q_a)>0 and len(q_b)>0 and q_a[-1]==q_b[0]) else q_a + q_b
            elif best_pattern == 'ascii_shift' and global_shift_val is not None:
                return "".join(chr(ord(c) + global_shift_val) for c in q_a)

        return None