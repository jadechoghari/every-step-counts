import random
import math
from typing import Dict, Any
import re
class MonteCarloTreeSearch():
    def __init__(self, c_puct=0.125, alpha=0.5, beta=0.9):
        # initialize parameters for MCTS
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta

    def _create_root_node(self, question: str) -> Dict[str, Any]:
        # ceate the root node for MCTS
        return {
            "state": question,
            "parent": None,
            "children": [],
            "visits": 0,
            "value": 0,
            "prior": 1.0,
        }

    def _select(self, node: Dict[str, Any]) -> Dict[str, Any]:
        # select a node based on UCB score\
        print("node children: ", node["children"])
        while node["children"]:
            node = max(node["children"], key=lambda c: self._ucb_score(c))
            print("node selected: ", node)
        return node

    def _ucb_score(self, node: Dict[str, Any]) -> float:
        # calculate UCB score for a node
        parent_visits = node["parent"]["visits"] if node["parent"] else 1
        return (node["value"] / (node["visits"] + 1)) + self.c_puct * node["prior"] * math.sqrt(parent_visits) / (1 + node["visits"])

    def _evaluate(self, node: Dict[str, Any]) -> float:
        # evaluate a node by expanding it or performing a rollout
        if self._is_terminal(node):
            print("its terminal: auto returing")
            return self._get_outcome_reward(node["state"])

        next_steps = self._generate_next_steps(node["state"])
        node["children"] = [self._create_child_node(node, step) for step in next_steps]
        print("node children not terminal ", node["children"])
        return self._rollout(node)

    def _create_child_node(self, parent: Dict[str, Any], step: str) -> Dict[str, Any]:
        # create a child node in the MCTS tree
        print("creating a child")
        return {
            "state": parent["state"] + " " + step,
            "parent": parent,
            "children": [],
            "visits": 0,
            "value": 0,
            "prior": self._get_prior(step),
        }

    def _rollout(self, node: Dict[str, Any]) -> float:
        # donduct a rollout simulation from a given node
        current_state = node["state"]
        #TODO: add L dynamically
        for _ in range(500):
            if self._is_terminal({"state": current_state}):
                break
            current_state += " " + self._choose_random_action(current_state)
        return self._get_outcome_reward(current_state)

    def _choose_random_action(self, state: str) -> str:
        # choose a random action during the rollout
        #TODO: implement a better random action selection strategy
        # print("Current state in choose random: ", state)
        prompt = f"Given the current state of solving a math problem: '{state}', what could be a possible next step? Provide a concise step."
        return self.generate_response(prompt)

    def _backpropagate(self, node: Dict[str, Any], value: float) -> None:
        # backpropagate the evaluation results up the tree
        while node:
            node["visits"] += 1
            node["value"] += value
            node = node["parent"]

    def _is_terminal(self, node: Dict[str, Any]) -> bool:
      # Check if a node is a terminal state by capturing "yes" or "no" in a phrase
      prompt = f"Is this a complete math solution? just output Yes or No: '{node['state']}'"
      print("Prompt in terminal: ", prompt)
      response = self.generate_response(prompt).strip().lower()
      print("response for is terminal: ", response)
      # Regular expressions to match variations of "yes" or "no"
      yes_pattern = re.compile(r'\byes\b', re.IGNORECASE)
      no_pattern = re.compile(r'\bno\b', re.IGNORECASE)
      
      
      if yes_pattern.search(response):
          print("is term is true")
          return True
      elif no_pattern.search(response):
          print("is term is false")
          return False
      
      # Fallback if neither "yes" nor "no" is found
      # You might want to log this case or handle it differently
      print("Response did not clearly indicate 'yes' or 'no'. Response was:", response)
      return True  # Default fallback, adjust based on your needs

