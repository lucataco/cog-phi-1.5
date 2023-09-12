# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-1_5"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True,
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKEN_CACHE,
            trust_remote_code=True,
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="Write a detailed analogy between mathematics and a lighthouse.\nAnswer:"),
        max_length: int = Input(description="max number of tokens", ge=0, le=1024, default=200),
    ) -> str:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=False
        ).to("cuda")
        outputs = self.model.generate(**inputs, max_length=max_length)
        text = self.tokenizer.batch_decode(outputs)[0]
        return text
