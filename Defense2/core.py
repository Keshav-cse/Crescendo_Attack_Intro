from dataclasses import dataclass, field
import time

try:
    from claim import JailBreakClass
    from llm_wrapper import LLMWrapper
except ImportError:
    from .claim import JailBreakClass
    from .llm_wrapper import LLMWrapper


SAFE_REFUSAL = (
    "I can't help with instructions for harmful or illegal actions. "
    "I can help with a safe, high-level explanation or defensive guidance instead."
)


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ChatCore:
    def __init__(
        self,
        classifier: JailBreakClass | None = None,
        llm: LLMWrapper | None = None,
        window_size: int = 6,
        max_turns: int = 20,
    ):
        self.classifier = classifier or JailBreakClass()
        self.llm = llm
        self.window_size = window_size
        self.max_turns = max_turns
        self.history: list[Turn] = [
            Turn(
                "system",
                "You are a helpful assistant. Refuse harmful instructions and offer safe alternatives.",
            )
        ]

    def handle_user_message(self, user_text: str) -> dict:
        if user_text.strip().lower() in {"reset", "new conversation", "start over"}:
            self.history = [self.history[0]]
            return {
                "answer": "Okay, starting a new conversation.",
                "guard": {
                    "verdict": "SAFE",
                    "normalized_score": 0.0,
                    "jailbreak_prob": 0.0,
                    "safe_prob": 1.0,
                },
            }

        self._add_turn("user", user_text)
        window = self._window_for_classifier()
        guard = self.classifier.score(window)

        if guard["verdict"] == "BLOCK":
            answer = SAFE_REFUSAL
        else:
            if self.llm is None:
                self.llm = LLMWrapper()
            answer = self.llm.generate(self._messages_for_llm())
            self._add_turn("assistant", answer)

        return {
            "answer": answer,
            "guard": guard,
        }

    def _add_turn(self, role: str, content: str) -> None:
        self.history.append(Turn(role, content))
        if len(self.history) > self.max_turns:
            system_turns = [turn for turn in self.history if turn.role == "system"]
            other_turns = [turn for turn in self.history if turn.role != "system"]
            self.history = system_turns[:1] + other_turns[-(self.max_turns - 1):]

    def _window_for_classifier(self) -> list[dict]:
        turns = self.history[-self.window_size:]
        return [{"role": turn.role, "content": turn.content} for turn in turns]

    def _messages_for_llm(self) -> list[dict]:
        return [{"role": turn.role, "content": turn.content} for turn in self.history]
