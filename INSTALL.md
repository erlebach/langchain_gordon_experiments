# Environment setup
We first install poetry
```
curl -sSL https://install.python-poetry.org | python3.11 -
```

# Create the virtual environment
```
rm pyproject.lock
```
if the lock file exists. 

# Enter the coding environment
```
poetry shell
```
Type `poetry env info` to see where it is located. 
Type `which python` to confirm that you are running the python in the 
project environment. 

# Add langchain libraries via pip
```
pip install -U llama-cpp-python --no-cache-dir
```

# Run Zephyr
```
cd zephyr_experiments
python my_langchain_gpu.py
python my_langchain_cpu.py
```


