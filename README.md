# Installation

To install this repository, run the following command from the main scope:

```bash
pip install --no-cache-dir -e  .
```

# Project Description

This project is structured as follows:

- `vendor/ultralytics_upstream`: A cloned repository of the upstream Ultralytics project (latest tag).
- `src/ultralytics`: The inner directory of the upstream repository, containing its core functionality.
- `src/ultralytics_extension`: Custom modifications and extensions built on top of the upstream repository.