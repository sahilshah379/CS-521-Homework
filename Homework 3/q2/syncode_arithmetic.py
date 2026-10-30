from syncode import Syncode
import torch

grammar = r"""
start: operand " " operator " " operand
operand: /[0-9]+(\.[0-9]*)?/
operator: "+" | "-" | "*" | "/"
"""

prompt = """Given the input, please extract the two operands and the operator. Output in the format below where each operand is a floating point and the operator is either + - * /.
Input:What is 19.7 plus 1.15?
Equation:19.7 + 1.15
Input:What is 18. minus 6.75?
Equation:18.0 - 6.75
Input:What is 7.2 multiplied by 0.65?
Equation:7.2 * 0.65
Input:What is 99.9 divided by 5.25?
Equation:99.9 / 5.25
"""

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
syn_llm = Syncode(model="microsoft/phi-2", grammar=grammar, parse_output_only=True, max_new_tokens=20, device=device)

queries = ["What is 327. multiplied by 11.0?", "What is 45.1 plus 23.54?", "What is 120.4 divided by 4.0?", "What is 104.2 minus 39.7?"]

def run_syncode(textual_query):
    output = syn_llm.infer(prompt + f"Input:{textual_query}\nEquation:")[0]
    operand1, operator, operand2 = output.split()
    operand1 = float(operand1)
    operand2 = float(operand2)
    answer = float(eval(output))
    return [operand1, operand2], operator, answer

for q in queries:
    operands, operator, answer = run_syncode(q)
    print(q)
    print(f"{operands[0]} {operator} {operands[1]} = {answer}")
