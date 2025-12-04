# Deploying to PyPI

## Automatic Deployment (Recommended)

The package is automatically published to PyPI when you create a GitHub release:

1. Go to the repository's Releases page
2. Click "Create a new release"
3. Create a tag (e.g., `v0.1.0`)
4. Publish the release
5. The GitHub Action will automatically build and upload to PyPI

## Manual Deployment

```bash
pip install build twine 
python -m build
twine upload dist/*
```