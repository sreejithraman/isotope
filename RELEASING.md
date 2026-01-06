# Releasing

## Automated Release (Recommended)

1. Go to Actions → "Create Release" → Run workflow
2. Enter version (e.g., `0.2.0`)
3. Select release type
4. Click "Run workflow"

The workflow will:
- Generate release notes using AI (Gemini)
- Update `pyproject.toml` version
- Update `CHANGELOG.md`
- Create a GitHub Release
- Publish to TestPyPI and PyPI

## Setup (One-time)

Add `GOOGLE_API_KEY` to repository secrets:

1. Get API key from https://aistudio.google.com/apikey
2. Go to Settings → Secrets → Actions → New repository secret
3. Name: `GOOGLE_API_KEY`, Value: your key

## Manual Release (Fallback)

If automation fails:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit and push
4. Create release: `gh release create v0.2.0 --generate-notes`

## Verification

After release, check:
- TestPyPI: https://test.pypi.org/project/isotope-rag/
- PyPI: https://pypi.org/project/isotope-rag/
- Install: `pip install isotope-rag==X.Y.Z`
