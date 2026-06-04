import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    is_available = True
except ImportError:
    is_available = False

sus_words = ["hypothetical","imagine","suppose","what if","let's say",
            "assume","consider","pretend","what if","let us say","let us imagine",
            "let us suppose","let us assume","let us consider","let us pretend"]
dan_words = ["kill","bomb","narcotics","terrorism","assassinate","hack","exploit","weapon","drugs","terror"
             "attack","violence","abuse","harassment","discrimination","hate speech","cyberbullying","child exploitation","human trafficking",
             ]
manupilative = ["but you said","but you told","but you aggreed"]

def _score(text: str):
    text_lower = text.lower()
    score = 0.0
    for words in sus_words:
        if words in text_lower:
            score += 1
    for words in dan_words:
        if words in text_lower:
            score += 5
    for words in manupilative:
        if words in text_lower:
            score += 3
    return min(score/10,1.0)
 
def _cosine_sim(a: np.ndarray, b:np.ndarray):
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    if d == 0:
        return 0.0
    return float(np.dot(a,b)/d)

class BOSS:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use: bool = True,
        suspicious: float = 0.65,
        block:float = 0.8,
    ):
        self.use = use and is_available
        self.suspicious = suspicious
        self.block = block
        self.model = None

        if self.use:
            self.model = SentenceTransformer(embedding_model)


    def scorer(self, buffer: "TurnBuffer"):

        window = buffer.get_window()
        user_content = [t.content for t in window if t.role == "user"]

        if not user_content:
            return None 
        

        combined_text = " ".join(user_content)
        score = _score(combined_text)
        ds = 0.0
        df = 0.0 
        if self.use and len(user_content) > 1:
            embeddings = self.model.encode(user_content,convert_to_numpy=True)
            ds = _cosine_sim(embeddings[0], embeddings[1]) #why? 
            
            delta = []
            for i in range(1, len(embeddings)):
                delta.append(1 - _cosine_sim(embeddings[i-1], embeddings[i]))
            df = np.mean(delta)

        final = 0.5*score + 0.25*ds + 0.25*df
        if final > self.block:
            ver =  "block"
        elif final > self.suspicious and final <= self.block:
            ver =  "suspicious"
        else:
            ver = "safe"
        return self._result(final, ver, score, ds, df)

    score = scorer
    

    @staticmethod
    def _result(score,verdict,kw,drift,delta):
        return {
            "score": score,
            "verdict": verdict,
            "keywords": kw,
            "drift": drift,
            "delta": delta,
        }
