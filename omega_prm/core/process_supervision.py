from typing import List, Dict, Any
import re
import tqdm
import random

class ProcessSupervision:
    def collect_process_supervision(self, question: str, golden_answer: str, search_limit: int = 100) -> List[Dict[str, Any]]:
        # dollect process supervision signals
        root = self._create_root_node(question)
        annotations = []

        # Add tqdm progress bar to the loop
        for _ in tqdm(range(search_limit), desc="Collecting Process Supervision"):
            leaf = self._select(root)
            value = self._evaluate(leaf)
            self._backpropagate(leaf, value)

            if self._is_terminal(leaf):
                annotations.extend(self._extract_annotations(leaf))
                if len(annotations) >= search_limit:
                    break

        #TODO: return when search limit if reached, and make sure this function is correct
        return annotations[:search_limit]

    def _extract_annotations(self, leaf: Dict[str, Any]) -> List[Dict[str, Any]]:
        # extract annotations from a terminal node
        steps = self._binary_search_steps(leaf["state"])
        annotations = []
        for i, step in enumerate(steps):
            annotations.append({
                "step": i + 1,
                "content": step,
                "correctness": self._evaluate_step_correctness(step),
            })
        return annotations

    def _binary_search_steps(self, solution: str) -> List[str]:
        # split the solution into steps using binary search
        sentences = re.split(r'(?<=[.!?]) +', solution)
        steps = []

        def split_step(start: int, end: int):
            if end - start <= 1:
                steps.append(" ".join(sentences[start:end]))
                return
            mid = (start + end) // 2
            split_step(start, mid)
            split_step(mid, end)

        split_step(0, len(sentences))
        return steps

    def _evaluate_step_correctness(self, step: str) -> float:
        # evaluate the correctness of a given step
        # TODO: implement a better way to evaluate step correctness
        prompt = f"On a scale of 0 to 1, how correct does this step in a math solution seem? , just output a number'{step}'"
        response = self.generate_response(prompt)
        try:
            return float(response.strip())
        except ValueError:
            print("WARNING: Random answer generated: ", "for this step: ", step)
            return random.uniform(0, 1)

    def _generate_next_steps(self, state: str) -> List[str]:
        # generate possible next steps for a solution
        #TODO: implement a better way to generate next steps
        prompt = f"Given the current state of solving a math problem: '{state}', suggest 3 logical next steps, separated by newlines."
        response = self.generate_response(prompt)
        return [step.strip() for step in response.split('\n') if step.strip()]
