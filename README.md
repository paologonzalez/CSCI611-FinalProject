## Environment Setup

We use a Python virtual environment to ensure consistent dependencies across all team members.

### 1. Create a Virtual Environment

From the root of the project:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

* **Mac/Linux:**

```bash
source venv/bin/activate
```

* **Windows:**

```bash
venv\Scripts\activate
```

You should now see `(venv)` in your terminal.

---

### 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet, install manually (example):

```bash
pip install pyyaml torch torchvision optuna matplotlib
```

---

### 4. Saving Dependencies

After installing new packages, update the dependency list:

```bash
pip freeze > requirements.txt
```

This ensures all team members can reproduce the exact environment.

---

### ⚠️ Important Notes

* Do **NOT** commit the `venv/` folder to GitHub
* Only commit `requirements.txt`
* The virtual environment is machine-specific and should be recreated locally

---

## Dataset Split Configuration (YAML)

We use a YAML file to define which images belong to the training, validation, and test sets. This ensures:

* Reproducibility
* Consistent evaluation across models
* Fair comparison during hyperparameter tuning

---

### Generating the Split File

Run:

```bash
python scripts/generate_split_manifest.py \
    --dataset-root ../dataset \
    --output configs/data_split.yaml
```

---

### YAML File Structure

Example:

```yaml
dataset_root: ../dataset
seed: 42

splits:
  train:
    - path: Baked Potato/img1.jpeg
      label: Baked Potato
      class_idx: 0

  val:
    - path: Crispy Chicken/img2.jpeg
      label: Crispy Chicken
      class_idx: 1

  test:
    - path: Baked Potato/img3.jpeg
      label: Baked Potato
      class_idx: 0

class_to_idx:
  Baked Potato: 0
  Crispy Chicken: 1
```

---

## Loading the YAML in Python

We use **PyYAML** to read the configuration file.

### Example

```python
import yaml

with open("configs/data_split.yaml", "r") as f:
    config = yaml.safe_load(f)

train_data = config["splits"]["train"]
val_data = config["splits"]["val"]
test_data = config["splits"]["test"]
```

---

### Why Use `yaml.safe_load`?

* Prevents execution of arbitrary code
* Safer for loading configuration files
* Recommended over `yaml.load`

---

## Summary

* Use a virtual environment (`venv/`) for dependency isolation
* Track dependencies with `requirements.txt`
* Use a YAML file to define dataset splits
* Load YAML configs using `PyYAML` for integration into the training pipeline

---
