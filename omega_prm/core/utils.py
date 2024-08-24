import random

def _get_prior(step: str) -> float:
    # calculate the prior probability of a step being correct
    #TODO: implement a better way to calculate the prior probability
    prompt = f"Rate this math step from 0 to 1: '{step}'"
    response = self.generate_response(prompt)
    try:
        number = float(response.strip())
        return min(max(number, 0.0), 1.0)
    except ValueError:
        return random.uniform(0, 1)

def _get_outcome_reward(state: str) -> float:
    # compute the outcome reward of a solution state
    #TODO: implement a better way to compute the outcome reward
    prompt = f"On a scale of -1 to 1, how correct does this math solution seem? '{state}'"
    response = self.generate_response(prompt)
    try:
        return float(response.strip())
    except ValueError:
        return random.uniform(-1, 1)
