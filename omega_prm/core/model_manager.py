from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

class OmegaPRM:
    def __init__(self, model_name="OuteAI/Lite-Mistral-150M-v2-Instruct", c_puct=0.125, alpha=0.5, beta=0.9, L=500):
        # initialize model and tokenizer
        #TODO: allow user to specify model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # set model to evaluation mode
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate_response(self, message: str, temperature: float = 0.4, repetition_penalty: float = 1.1) -> str:
        # generate a response using the model
        #TODO: fix max_length token issue and make sure LLM is outputting accurate responses
        input_ids = self.tokenizer.encode(message, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=1024,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.strip()

    def _get_prior(self, step: str) -> float:
      # calculate the prior probability of a step being correct
      #TODO: implement a better way to calculate the prior probability
      prompt = f"Rate this math step from 0 to 1: '{step}'"
      response = self.generate_response(prompt)
      try:
          number = float(response.strip())
          return min(max(number, 0.0), 1.0)
      except ValueError:
          return random.uniform(0, 1)

    def _get_outcome_reward(self, state: str) -> float:
        # compute the outcome reward of a solution state
        #TODO: implement a better way to compute the outcome reward
        prompt = f"On a scale of -1 to 1, how correct does this math solution seem? '{state}'"
        response = self.generate_response(prompt)
        try:
            return float(response.strip())
        except ValueError:
            return random.uniform(-1, 1)
