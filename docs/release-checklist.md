# Release checklist (0.2.0)

Maintainer-run steps for publishing ragsearch. Claude Code prepares PRs and commands; **only the
maintainer tags, publishes or changes repository settings.** Publishing is done by
`.github/workflows/python-publish.yml` from a GitHub Release:

| GitHub Release | Version in `pyproject.toml` | Tag | Target commit | Publishes to |
|---|---|---|---|---|
| **pre-release** | candidate, e.g. `0.2.0rc1` | `v0.2.0rc1` | on `develop` (or `master`) | TestPyPI |
| normal release | final, e.g. `0.2.0` | `v0.2.0` | on `master` | PyPI |

The workflow refuses to build if the tag isn't `v` + the `pyproject.toml` version, if the pre-release
flag doesn't match the version, or if the tagged commit isn't on an allowed branch. Commands below
use the GitHub CLI (`gh`) and assume `REPO=mrutunjay-kinagi/ragsearch`.

## 1. Before the release candidate

- [ ] Every 0.2.0 issue except #109 is closed: `gh issue list --repo "$REPO" --milestone 0.2.0 --state open`
- [ ] CI is green on `develop`: `gh run list --repo "$REPO" --branch develop --workflow ci.yml --limit 1`
- [ ] The version-bump PR (`0.2.0rc1`, CHANGELOG `0.2.0` section) is merged into `develop`.
- [ ] Repository secrets `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN` exist:
      `gh api repos/$REPO/actions/secrets --jq '.secrets[].name'`

## 2. Publish `0.2.0rc1` to TestPyPI (rehearsal)

```bash
REPO=mrutunjay-kinagi/ragsearch
gh release create v0.2.0rc1 --repo "$REPO" --target develop --prerelease \
  --title "ragsearch 0.2.0rc1" \
  --notes "Release candidate for 0.2.0, published to TestPyPI for a rehearsal. Not for production use."

# Follow the publish run: "Check release, test and build" and "Publish to TestPyPI" must pass;
# "Publish to PyPI" must be skipped.
gh run watch --repo "$REPO" "$(gh run list --repo "$REPO" --workflow python-publish.yml --limit 1 --json databaseId -q '.[0].databaseId')"

# The candidate is on TestPyPI:
curl -s https://test.pypi.org/pypi/ragsearch/json | python3 -c "import json,sys; print(sorted(json.load(sys.stdin)['releases']))"
```

## 3. Install the candidate in a fresh virtual environment

Dependencies aren't on TestPyPI, so PyPI is added as an extra index. **Pin the exact version:**
pip skips pre-releases unless pinned, and PyPI has `ragsearch` 0.1.x, so an unpinned install would
quietly install 0.1.5 from PyPI.

```bash
python3.12 -m venv ~/ragsearch-rc && source ~/ragsearch-rc/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ragsearch==0.2.0rc1"

python -c "import importlib.metadata as m, ragsearch; print(m.version('ragsearch'), ragsearch.__file__)"  # 0.2.0rc1, site-packages
python -m pip show chromadb   # must report "Package(s) not found" (chromadb is an optional extra)
```

Safer alternative (avoids resolving dependency names against TestPyPI, where anyone can upload a
same-named package): install only ragsearch from TestPyPI, then its dependencies from PyPI.

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps "ragsearch==0.2.0rc1"
python -m pip install "ragsearch==0.2.0rc1"   # already installed; resolves its dependencies from PyPI
```

## 4. Check the candidate with the real API

```bash
export COHERE_API_KEY="your-cohere-api-key"

# Quickstart, exactly as documented (in an empty folder):
mkdir -p ~/ragsearch-rc-quickstart && cd ~/ragsearch-rc-quickstart
curl -sL https://raw.githubusercontent.com/mrutunjay-kinagi/ragsearch/v0.2.0rc1/docs/quickstart.md \
  | python -c "import re,sys; print(re.findall(r'^\`\`\`python[^\n]*\n(.*?)^\`\`\`', sys.stdin.read(), re.S | re.M)[0])" > quickstart.py
python quickstart.py   # "Indexed 8 claims." and a cited answer about the windshield claim

# Every README example, in order, against the installed package (run from a clone of the tag,
# so the sample files exist; the script refuses to run against repository code):
git clone --depth 1 --branch v0.2.0rc1 https://github.com/mrutunjay-kinagi/ragsearch.git ~/ragsearch-rc-src
cd ~/ragsearch-rc-src && python .github/ci/run_readme_examples.py
# Optional: OPENAI_API_KEY=... / RUN_OLLAMA=1 / RUN_LMSTUDIO=1 enable the other provider examples.
```

- [ ] Quickstart answers with a citation.
- [ ] `run_readme_examples.py`: `0 failed`.
- [ ] Web server by hand: `rag_engine.run()` then `threading.Event().wait()` in a script; `curl -X POST
      127.0.0.1:8080/answer -H 'Content-Type: application/json' -d '{"query": "Why was the windshield claim denied?"}'`.
- [ ] The claim documents in the local `dataset/` answer with default `setup()` / `answer()`.

## 5. After the rehearsal passes

- [ ] **Swap the TestPyPI token for a project-scoped one:** on TestPyPI, create an API token scoped
      to the `ragsearch` project, then `gh secret set TEST_PYPI_API_TOKEN --repo "$REPO"` (paste it),
      and delete the account-scoped token in the TestPyPI account settings.
- [ ] Final version PR on `develop` (prepared by Claude Code): `0.2.0rc1` → `0.2.0`; the CHANGELOG
      `0.2.0` heading gets the release date; remove the "newer than the latest PyPI release … install
      from GitHub" notes from the README and quickstart.
- [ ] Release PR from `develop` into `master` (opened by Claude Code), with CI green.

## 6. Release `0.2.0` to PyPI (maintainer)

```bash
REPO=mrutunjay-kinagi/ragsearch
# 1. Merge the release PR into master with a regular merge commit (in GitHub, or):
gh pr merge <release-PR-number> --repo "$REPO" --merge

# 2. Release notes = the CHANGELOG's 0.2.0 section, then tag and publish from master:
git fetch origin && git show origin/master:CHANGELOG.md \
  | python3 -c "import sys; t=sys.stdin.read(); s=t.index('## [0.2.0]'); e=t.find('\n## [', s+1); print(t[s:e if e>0 else None].split('\n',1)[1].strip())" > /tmp/ragsearch-0.2.0-notes.md
gh release create v0.2.0 --repo "$REPO" --target master --title "ragsearch 0.2.0" --notes-file /tmp/ragsearch-0.2.0-notes.md --latest

# 3. Follow the publish run: "Publish to PyPI" must pass; "Publish to TestPyPI" must be skipped.
gh run watch --repo "$REPO" "$(gh run list --repo "$REPO" --workflow python-publish.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

- [ ] `https://pypi.org/project/ragsearch/0.2.0/` shows the new description, metadata and README.
- [ ] Fresh venv: `python -m pip install ragsearch==0.2.0`, then the quickstart runs.
- [ ] The CI badge on `master` shows the workflow status.
- [ ] Merge `master` back into `develop` so the release merge commit is on both.
- [ ] Close the 0.2.0 milestone and #109.
- [ ] Later (0.2.1): Trusted Publishing instead of tokens (#110).
