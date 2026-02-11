# Detectron V2 tuning

## Installation

This model requires special workflow. Model must be installed directly from external repo
as a separate dependency.

```shell
# install normal dependencies
uv sync

# install detectron but when building source - reuse dependencies from step 1
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```
