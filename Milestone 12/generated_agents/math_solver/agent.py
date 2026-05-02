
import os
from fastapi import FastAPI
from numexpr import evaluate
from sympy import sympify

app = FastAPI()

@app.post("/run")
def run(input_data: dict):
    try:
        math_problem = input_data['input']
        result = evaluate(math_problem)
        return {"output": str(result)}
    except Exception as e:
        return {"output": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "agent": "math_solver"}
