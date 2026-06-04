import fcntl
import os
import sys

try:
    from core import ChatCore
except ImportError:
    from .core import ChatCore


LOCK_PATH = "/tmp/defense2-main.lock"


def acquire_single_instance_lock():
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Defense2.main is already running. Close the old terminal first.")
        sys.exit(1)

    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def main():
    lock_file = acquire_single_instance_lock()
    chat = ChatCore()
    print("Type 'exit' or 'quit' to stop.")

    try:
        while True:
            user_text = input("\nYou: ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            result = chat.handle_user_message(user_text)
            guard = result["guard"]
            print(f"Guard: {guard['verdict']} | score={guard['normalized_score']:.3f}")
            print(f"Assistant: {result['answer']}")
    finally:
        lock_file.close()


if __name__ == "__main__":
    main()
