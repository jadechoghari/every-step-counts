## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/every-step-counts.git
cd every-step-counts
pip install -r requirements.txt

## How to use

```python
from omega_prm.core.model_manager import OmegaPRM
from omega_prm.core.process_supervision import ProcessSupervision

# start the model
omega_prm = OmegaPRM(model_name="OuteAI/Lite-Mistral-150M-v2-Instruct")

# define your math problem and the golden answer
question = "What is the sum of the first 100 positive integers?"
golden_answer = "5050"

# create a ProcessSupervision instance and collect annotations
process_supervision = ProcessSupervision()
annotations = process_supervision.collect_process_supervision(question, golden_answer, search_limit=100)

# display the first few annotations
print(annotations[:5])
```