"""Terminal chat for the demo:  python -m backend.cli"""
import uuid
from .engine import FactoryLens

if __name__ == "__main__":
    bot, sid = FactoryLens(), str(uuid.uuid4())
    print("FactoryLens AI (synthetic data). Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if q.strip().lower() in {"quit", "exit"}:
            break
        r = bot.ask(q, sid)
        print(f"\nFactoryLens AI:\n{r['reply']}")
        if r.get("quick_replies"):
            print("Quick replies: " + " | ".join(r["quick_replies"]))
        print()
