# Publish a release

Releases should contain the same distributions that passed validation, without
placing a long-lived upload credential in the repository. The
[`publish-pypi.yml` workflow](../.github/workflows/publish-pypi.yml) builds the
wheel and source distribution, then publishes them through PyPI trusted publishing.

## Prepare the release

Choose a version that has not already been published, update `pyproject.toml` and
the [changelog](../CHANGELOG.md), and refresh `uv.lock` with `uv lock`. Commit and
push the changes to `main`, then wait for the full
[quality workflow](../.github/workflows/ci.yml) to pass. Run the
[local verification steps](../README.md#development-and-artifact-verification)
before pushing as well.

## Configure trust

The PyPI project `langchain-ollie` must trust GitHub owner `mylonasc`, repository
`langchain-ollie`, and workflow file `publish-pypi.yml`. The workflow currently
uses no GitHub environment, so leave the publisher's environment field empty.
No PyPI password or API-token secret is needed. If you add an environment later,
configure the same name in both the workflow and the PyPI publisher.

Only the publishing job can request an OIDC token. That job downloads the built
distributions without checking out or executing project code. The PyPA publishing
action validates the package metadata and uploads provenance attestations along
with the distributions.

## Start publication

For a manual release, dispatch the workflow from `main` with the exact version:

```bash
gh workflow run publish-pypi.yml --ref main -f version=0.1.0
```

Alternatively, publish a GitHub release whose tag is `vVERSION` or `VERSION`,
matching the version in `pyproject.toml` at that tag. Version mismatches fail
before upload. The manual path is restricted to `main`; neither path publishes
on ordinary pushes or pull requests.

PyPI releases are immutable: do not reuse a version for different artifacts.
If an upload fails, inspect the workflow logs and PyPI's file listing before
retrying; a partial upload may have succeeded. The workflow does not silently
skip duplicate files.

## Confirm the release

Check that both the wheel and source distribution appear on the
[PyPI release page](https://pypi.org/project/langchain-ollie/), then install the
exact version in a fresh Python 3.12–3.14 environment outside the checkout:

```bash
python -m pip install --index-url https://pypi.org/simple 'langchain-ollie==0.1.0'
python -c 'from langchain_ollie import RecordingSession; print(RecordingSession)'
python -m langchain_ollie --help
```

Use the version being released in place of `0.1.0` for later releases. Keep the
workflow URL with the release record so the uploaded files can be traced back to
the source commit and validation run.
