from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
import re

class OmegaPRM:
    def __init__(self, model_name="OuteAI/Lite-Mistral-150M-v2-Instruct", c_puct=0.125, alpha=0.5, beta=0.9, L=500):
        # initialize model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # use the pipeline for text generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,  # use CUDA if available
            model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
        )
        
        self.temperature = 0.4
        self.repetition_penalty = 1.1

    def generate_response(self, message: str, temperature: float = 0.4, repetition_penalty: float = 1.1) -> str:
        # generate a response using the pipeline
        messages = [{"role": "user", "content": message}]
        
        # generate the response using the pipeline
        outputs = self.pipe(messages, max_new_tokens=256)
        generated_text = outputs[0]['generated_text'][-1]['content'].strip()
        
        print("generated_text", generated_text)
        return generated_text

    def _get_prior(self, step: str) -> float:
    # calculate the prior probability of a step being correct
      # print("get prior step: ", step)
      prompt = f"Rate this math step from 0 to 1: '{step}'"
      response = self.generate_response(prompt).strip()
      
      # a regular expression to extract a number from the response
      number_pattern = re.compile(r'[-+]?\d*\.\d+|\d+', re.IGNORECASE)
      match = number_pattern.search(response)
      
      if match:
          try:
              number = float(match.group())
              return min(max(number, 0.0), 1.0)
          except ValueError:
              return random.uniform(0, 1)
      else:
          # Fallback if no number is found
          print("No numeric value found in response. Response was:", response)
          return random.uniform(0, 1)

    
    def _get_outcome_reward(self, state: str) -> float:
        # we compute the outcome reward of a solution state
        # print("In get outcome reward: ", state)
        prompt = f"On a scale of -1 to 1, how correct does this math solution seem? '{state}'"
        response = self.generate_response(prompt).strip()
        
        # this is a regular expression to extract a number from the response
        number_pattern = re.compile(r'[-+]?\d*\.\d+|\d+', re.IGNORECASE)
        match = number_pattern.search(response)
        
        if match:
            try:
                number = float(match.group())
                return min(max(number, -1.0), 1.0)
            except ValueError:
                return random.uniform(-1, 1)
        else:
            # Fallback if no number is found
            print("No numeric value found in response. Response was:", response)
            return random.uniform(-1, 1) 
