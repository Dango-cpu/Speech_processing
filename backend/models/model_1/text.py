import re
import unicodedata

BLANK_TOKEN = "<blank>"
VIETNAMESE_CHARACTERS = (
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "bcdđ"
    "eèéẻẽẹêềếểễệ"
    "fghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợ"
    "pqrstuùúủũụưừứửữự"
    "vwxyỳýỷỹỵz"
    " '"
)
DEFAULT_VOCAB = [BLANK_TOKEN] + list(dict.fromkeys(VIETNAMESE_CHARACTERS))


class TextTransform:
    def __init__(self, vocab=None):
        self.vocab = vocab or DEFAULT_VOCAB
        self.char_to_idx = {char: index for index, char in enumerate(self.vocab)}
        self.idx_to_char = {index: char for index, char in enumerate(self.vocab)}
        self.blank_index = self.char_to_idx[BLANK_TOKEN]
        allowed_chars = "".join(
            re.escape(char)
            for char in self.vocab
            if char != BLANK_TOKEN
        )
        self.allowed_pattern = re.compile(f"[^{allowed_chars}]+")

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text.lower().replace("\u2019", "'"))
        text = self.allowed_pattern.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def encode(self, text):
        normalized = self.normalize(text)
        return [self.char_to_idx[char] for char in normalized if char in self.char_to_idx]

    def decode_greedy(self, token_ids):
        collapsed = []
        previous = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != previous and token_id != self.blank_index:
                collapsed.append(self.idx_to_char.get(token_id, ""))
            previous = token_id
        return "".join(collapsed).strip()

    def to_metadata(self):
        return {"vocab": self.vocab, "blank_index": self.blank_index}
