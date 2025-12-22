# scripts/run_chat.py
from src.inference import ask_final

while True:
    msg = input("You : ")
    if msg.lower() in {"exit", "quit"}:
        break

    reply = ask_final(msg)
    print(f"output: {reply}")
