
import re
from typing import List, Tuple, Dict, Optional

from src.solvers.equations.ast_brute_force import ASTBruteForceSolver
from src.solvers.equations.pseudo_math import PseudoMathStringSolver
from src.solvers.equations.cryptarithm import CryptarithmCSPSolver

class BaseEquationSolver:
    """
    Базовый класс-маршрутизатор. 
    Отвечает за парсинг сырого промпта, эвристическую классификацию задачи 
    и передачу управления соответствующему суб-солверу.
    """
    def __init__(self):
        # Инициализация пула доступных солверов
        self.ast_solver = ASTBruteForceSolver()
        self.csp_solver = CryptarithmCSPSolver()
        self.string_solver = PseudoMathStringSolver()

    def _extract_sections(self, text: str) -> Tuple[str, str]:
        pattern = r"few examples:?\s*(.*?)\s*now, d[e]?termine the result for:\s*(.*)"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", ""

    def _classify_task(self, examples: str, target: str) -> str:
        if not examples.strip():
            return "Empty"
        
        # 1. Cryptarithms (базовая проверка)
        if not re.search(r'\d', examples):
            # ВАЖНО: Если цифр нет, это может быть Cryptarithm, 
            # НО сначала нужно исключить тупую склейку!
            # Поэтому логику склейки лучше вынести ДО Cryptarithm, 
            # либо проверять её внутри этого блока.
            pass # Переходим к проверке на склейку ниже
        
        # 2. Pseudo-Math (строго по целевому оператору)
        # Ищем первый не-буквенно-цифровой символ, исключая пробелы и '='
        target_op_match = re.search(r'[^\w\s=]', target)
        
        if target_op_match:
            target_op = target_op_match.group(0)
            
            for line in examples.split('\n'):
                if '=' in line and target_op in line.split('=')[0]:
                    lhs, rhs = line.split('=', 1)
                    
                    # Разбиваем строку строго по оператору и убираем лишние пробелы
                    operands = [op.strip() for op in lhs.split(target_op)]
                    rhs_clean = rhs.strip()
                    
                    if len(operands) == 2:
                        # Прямая склейка (A + B = AB)
                        if rhs_clean == "".join(operands):
                            return "Pseudo-Math (Format/String)"
                        
                        # Реверсивная склейка операндов (A + B = BA)
                        if rhs_clean == "".join(reversed(operands)):
                            return "Pseudo-Math (Format/String)"
                        
                        # Полный посимвольный реверс прямой склейки (редко, но бывает: AB -> BA)
                        if rhs_clean == "".join(operands)[::-1]:
                            return "Pseudo-Math (Format/String)"

        # Если цифр вообще нет и это не склейка — отправляем в перебор шифров
        if not re.search(r'\d', examples):
            return "Cryptarithm (CSP)"

        # 3. Все остальное
        return "AST Brute-force"

    def solve(self, prompt: str) -> Optional[str]:
        examples, target = self._extract_sections(prompt)
        if not examples or not target:
            return None
            
        task_type = self._classify_task(examples, target)
        
        # Подключение специфичного солвера в зависимости от класса задачи
        if task_type == "AST Brute-force":
            return self.ast_solver.solve(examples, target)
        elif task_type == "Cryptarithm (CSP)":
            #return self.csp_solver.solve(examples, target)
            return "HUI"
        elif task_type == "Pseudo-Math (Format/String)":
            return self.string_solver.solve(examples, target)
            
        return None