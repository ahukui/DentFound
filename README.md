## Installation

> **⚠️ Important**  
> These instructions are for **Linux only**.

---

### 1. Clone the Repository

```bash
git clone https://github.com/ahukui/DentFound.git
cd DentFound
```

### 2. Create and Activate Conda Environment

```bash
conda create -n dentfound python=3.10 -y
conda activate dentfound
```

### 3. Install Core Package

```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### 4. Install Training Dependencies (Optional)

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 5. Upgrade to the Latest Codebase

```bash
git pull
pip install -e .
```

> **⚠️ Important**  
> If you encounter import errors after upgrading, try reinstalling flash-attn:
> ```bash
> pip install flash-attn --no-build-isolation --no-cache-dir
> ```
