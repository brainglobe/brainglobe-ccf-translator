# Deploying to PyPI

## Automatic Deployment (Recommended)

The package is automatically published to PyPI when you push a version tag:

1. Update your version as needed
2. Create and push a tag: `git tag v0.1.0 && git push --tags`
3. The GitHub Action will run tests, build, and upload to PyPI

**Note:** You need to add a `TWINE_API_KEY` secret to your repository with your PyPI API token.

## Manual Deployment

```bash
pip install build twine 
python -m build
twine upload dist/*
```