import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

Model = Path("./model")

class JailBreakClass:

    def __init__(self,model_dir: str = str(Model), sus_threshold : float = 0.6, block: float = 0.8):
        self.sus_threshold = sus_threshold
        self.block = block
        self.device = "cpu"

        self.block_keywords = [
            "make a bomb",
            "build a bomb",
            "detonate",
            "explosives",
            "weaponize",
            "poison",
            "kill yourself",
            "self-harm",
            "bypass safety",
            "jailbreak",
            "harmful instructions",
        ]

        self.crescendo_indicators = [
            "continue",
            "continuing",
            "building on",
            "build on",
            "from before",
            "earlier",
            "again",
            "now explain",
            "hypothetically",
            "for a story",
            "roleplay",
            "pretend",
            "ignore previous",
            "as a continuation",
            "what do you think",
            "how was it made back then",
        ]

        self.crescendo_similarity_threshold = 0.55
        self.crescendo_block_similarity = 0.75

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score(self, con: list[dict]):
        latest_user_text = self._latest_user_message(con)
        text = latest_user_text

        if self._contains_block_keyword(latest_user_text):
            return {
                "normalized_score": 1.0,
                "verdict": "BLOCK",
                "jailbreak_prob": 1.0,
                "safe_prob": 0.0,
                "reason": "rule_based_block",
            }

        prior_harmful_texts = self._harmful_user_messages(con)
        if self._is_crescendo_attempt(latest_user_text) and prior_harmful_texts:
            return {
                "normalized_score": 1.0,
                "verdict": "BLOCK",
                "jailbreak_prob": 1.0,
                "safe_prob": 0.0,
                "reason": "crescendo_block",
            }

        similarity = self._similarity_to_prior_harmful(latest_user_text, prior_harmful_texts)
        if similarity >= self.crescendo_block_similarity:
            return {
                "normalized_score": similarity,
                "verdict": "BLOCK",
                "jailbreak_prob": similarity,
                "safe_prob": 1.0 - similarity,
                "reason": "dangerous_direction",
            }
        if similarity >= self.crescendo_similarity_threshold:
            return {
                "normalized_score": similarity,
                "verdict": "SUS",
                "jailbreak_prob": similarity,
                "safe_prob": 1.0 - similarity,
                "reason": "dangerous_direction",
            }

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()
        happy = float(probs[0])
        jail = float(probs[1])

        if jail >= self.block:
            verdict = "BLOCK"
        elif jail >= self.sus_threshold:
            verdict = "SUS"
        else:
            verdict = "SAFE"
 
        return {
            "normalized_score": jail,
            "verdict": verdict,
            "jailbreak_prob": jail,
            "safe_prob": happy,
        }
    
    def _contains_block_keyword(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.block_keywords)

    def _is_crescendo_attempt(self, text: str) -> bool:
        text_lower = text.lower()
        return any(signal in text_lower for signal in self.crescendo_indicators)

    def _harmful_user_messages(self, window: list[dict]) -> list[str]:
        return [
            turn.get("content", "")
            for turn in window[:-1]
            if turn.get("role") == "user" and self._contains_block_keyword(turn.get("content", ""))
        ]

    def _similarity_to_prior_harmful(self, latest_user_text: str, harmful_texts: list[str]) -> float:
        if not latest_user_text or not harmful_texts:
            return 0.0

        texts = harmful_texts + [latest_user_text]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            tfidf = vectorizer.fit_transform(texts)
            scores = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
            return float(scores.max()) if len(scores) else 0.0
        except ValueError:
            return 0.0

    def _latest_user_message(self, window: list[dict]) -> str:
        for turn in reversed(window):
            if turn.get("role") == "user":
                return turn.get("content", "")
        return ""

    def _format_window(self, window: list[dict]) -> str:
        parts = []
        for turn in window:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return " | ".join(parts)

 

 



