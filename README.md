# 🤖 AgentBeats-EgoErrorVQA







## 🎞 Video Data Source

You need to download the video data first.

- [CaptainCook4D](https://captaincook4d.github.io/captain-cook/)

- [Epic-Tent](https://sites.google.com/view/epic-tent)
(ONLY download 02,04,05,06 mp4 video will be enough.)

- [EgoOops](https://y-haneji.github.io/EgoOops-project-page/)

- [Assembly101](https://assembly101.github.io/)
(ONLY download the egocentric video will be enough.)

## 💻 Organize Video Paths

We use the same naming and organization method for datasets from the same source; 
simply place the downloaded datasets together in one folder.

```bash
VIDEO_PATH_PATTERNS = {
        "captaincook4d": [
            "{video_root}/{video_subdir}/{video_id}_360p.mp4",
        ],
        "epic_tent": [
            "{video_root}/{video_subdir}/0{video_id}.tent.gopro.MP4",
        ],
        "assembly101": [
            "{video_root}/{video_subdir}/{video_id}/HMC_84355350_mono10bit.mp4",
            "{video_root}/{video_subdir}/{video_id}/HMC_21110305_mono10bit.mp4",
        ],
        "egooops": [
            "{video_root}/{video_subdir}/{task_id}/{video_id}.MP4",
        ]
    }
```



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
