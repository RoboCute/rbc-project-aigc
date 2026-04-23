# AIGC Robocute Project Template

This is the template project for [**Robocute**](https://github.com/RoboCute/RoboCute)

for AIGC usage

uv sync --extra cu128
uv run modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir ./pretrained/zimage
uv run main.py