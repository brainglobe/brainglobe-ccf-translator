to deploy the package to PyPi


```bash
pip install build twine 
python -m build
twine upload dist/*
```