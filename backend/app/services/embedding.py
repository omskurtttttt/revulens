"""
RevuLens DistilBERT Embedding Extractor.

Extracts fixed-dimensional contextual representations using frozen
'distilbert-base-multilingual-cased' per GEMINI.md.

Design constraints:
- Encoder is FROZEN (no fine-tuning).
- Default pooling: Mean pooling over token vectors excluding padding.
- Optional pooling: CLS token vector (reported alongside mean pooling).
- Max sequence length: 128-256 tokens.
- Supports both PyTorch device acceleration (CUDA / CPU) and batch processing.
"""

from typing import List, Union, Optional
import numpy as np

# Lazy imports for torch and transformers so module can be imported cleanly
_torch = None
_transformers = None


def _lazy_init():
    global _torch, _transformers
    if _torch is None or _transformers is None:
        import torch
        import transformers
        _torch = torch
        _transformers = transformers


class DistilBERTEmbeddingExtractor:
    """Frozen DistilBERT multilingual embedding extractor with mean and CLS pooling."""

    DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
        pooling_strategy: str = "mean",
        device: Optional[str] = None
    ):
        _lazy_init()
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy.lower()

        if self.pooling_strategy not in ("mean", "cls"):
            raise ValueError(f"Unsupported pooling strategy '{pooling_strategy}'. Choose 'mean' or 'cls'.")

        if device is None:
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = _transformers.DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = _transformers.DistilBertModel.from_pretrained(self.model_name)
        
        # Strictly freeze all encoder weights per GEMINI.md
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def _mean_pooling(self, token_embeddings: "_torch.Tensor", attention_mask: "_torch.Tensor") -> "_torch.Tensor":
        """
        Mean pooling over token vectors, excluding padding tokens.
        
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Expand attention mask: (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = _torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = _torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _cls_pooling(self, token_embeddings: "_torch.Tensor") -> "_torch.Tensor":
        """
        CLS token representation (first token vector).
        
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        return token_embeddings[:, 0, :]

    def extract(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        pooling_strategy: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract frozen DistilBERT embeddings for a string or list of strings.
        
        Returns:
            numpy.ndarray of shape (N, 768)
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.empty((0, 768), dtype=np.float32)

        strategy = (pooling_strategy or self.pooling_strategy).lower()
        embeddings_list = []

        with _torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # Defensive check for None or empty strings
                batch_texts = [t if (t and isinstance(t, str)) else "" for t in batch_texts]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)

                if strategy == "mean":
                    pooled = self._mean_pooling(last_hidden_state, attention_mask)
                elif strategy == "cls":
                    pooled = self._cls_pooling(last_hidden_state)
                else:
                    raise ValueError(f"Unknown pooling strategy: {strategy}")

                embeddings_list.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(embeddings_list)
