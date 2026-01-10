# AgentBeats-EgoErrorVQA







## Video Data Source

CaptainCook4D

Epic-Tent
(ONLY download 02,04,05,06 mp4 video will be enough.)
EgoOops

Assembly101
(ONLY download the egocentric video will be enough.)


## 🚀 Quick start

### Installation

```bash
# Use UV to synchronize dependencies
uv sync
```


### Usage
Carefully modify the data path and change the white agent format code for your own agent. (This code uses Qwen2.5-VL as an example.)

```bash
# Launch complete evaluation
uv run python main.py launch

# Only perform open-end evaluation
uv run python main.py launch --task-type  open-end

# Only perform multiple-choice evaluation
uv run python main.py launch --task-type  multiple-choice
```

In ```main.py```， you can view customizable settings, including selecting the dataset you want to test.


### Only run White Agent (Qwen2.5-VL)

```bash
# Default configuration (port 9002, CUDA:0)
uv run python -m src.white_agent.agent

# Custom configuration
uv run python -c "
from src.white_agent.agent import start_white_agent
start_white_agent(
    model_path='/path/to/Qwen2.5-VL-7B-Instruct',
    device='cuda:0',
    host='localhost',
    port=9002
)
"
```
