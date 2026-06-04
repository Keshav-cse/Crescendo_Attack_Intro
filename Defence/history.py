from dataclasses import dataclass,fields,field
from datetime import datetime 
from typing import Literal
import time 


Role = Literal["User","Assistant","System"]

@dataclass
class turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=lambda: time.time())
    flagged_dangerous:bool = False
    escalated_score:float = 0.0
class TurnBuffer:
    def __init__(self,max_turns: int = 20, window_size: int = 10):
        self.max_turns = max_turns
        self.window_size = window_size
        self.turns: list[turn] = []
        self.reset: int = 0
    def add_turn(self,role:Role,content:str) -> list[turn]:
        if len(self.turns) >= self.max_turns:
            self.turns.remove(self.turns[0])
        self.turns.append(turn(role,content))
        return self.turns
    def reset_turns(self,keepHistory: bool = True):
        if keepHistory:
            self.turns = [t for t in self.turns if t.role == "System"] 
        else:
            self.turns = []
        self.reset += 1
    def get_window(self):
        return self.turns[-self.window_size:]
    def user_turns(self):
        return [t for t in self.turns if t.role == "User"]

    def flag_turn(self,turn: Turn, score: float):
        turn.flagged_dangerous = True
        turn.escalated = score
