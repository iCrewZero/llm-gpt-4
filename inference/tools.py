def calculator(expr: str):
    try:
        return str(eval(expr))
    except:
        return "error"

TOOLS = {
    "calculator": calculator
}
