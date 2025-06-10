# TopFarm Development Guide

This document provides guidelines and information for developers working on the TopFarm project. It covers setting up the development environment, running tests, understanding the CI/CD pipeline, and other relevant practices.

## 1. Introduction

TopFarm is a wind farm optimization tool. Development relies on Python, `pixi` for environment and task management, and GitLab CI for continuous integration and deployment.

Key files and directories:
-   `pyproject.toml`: Defines project metadata, dependencies, build system, and `pixi` configurations.
-   `.gitlab-ci.yml`: Defines the CI/CD pipeline jobs.
-   `topfarm/`: Main source code directory.
-   `docs/`: Documentation source files.
-   `examples/`: Example scripts and use cases. This directory is deprecated and contains irrelevant code.
-   `README.md`: Project overview and links.

## 2. Setting up the Development Environment with Pixi

`pixi` is used to manage project dependencies and environments. The configuration for `pixi` can be found in the `pyproject.toml` file under the `[tool.pixi.*]` sections.

**Installation:**
If you don't have `pixi` installed, follow the instructions on the [official pixi website](https://pixi.sh/latest/).

**Initializing the Environment:**
To set up the development environment for TopFarm:
1.  Clone the repository:
    ```sh
    git clone https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git
    cd TopFarm2
    ```
2.  Install dependencies using `pixi`. This will create and populate a `.pixi` directory with the environment.
    ```sh
    pixi install
    ```
    Or, to activate a specific environment (e.g., for Python 3.11):
    ```sh
    pixi shell -e py311
    ```
    The available environments are defined in `[tool.pixi.environments]` in [`pyproject.toml`](pyproject.toml):
    - `py39`
    - `py310`
    - `py311`
    - `py312`

**Editable Install:**
The [`pyproject.toml`](pyproject.toml) file specifies an editable install for TopFarm within the `pixi` environment under `[tool.pixi.pypi-dependencies]`:
```toml
[tool.pixi.pypi-dependencies]
topfarm = { path = ".", editable = true }
```
This means changes to the source code are immediately reflected in the environment.

## 3. Running Tests with Pixi

Tests are run using `pytest`. `pixi` tasks are defined in [`pyproject.toml`](pyproject.toml) to simplify running tests across different Python versions.

**Test Task Definitions (from [`pyproject.toml`](pyproject.toml)):**
-   Individual Python versions:
    -   `test39 = "pytest -n auto"` (for Python 3.9 environment)
    -   `test310 = "pytest -n auto"` (for Python 3.10 environment)
    -   `test311 = "pytest -n auto"` (for Python 3.11 environment)
    -   `test312 = "pytest -n auto"` (for Python 3.12 environment)
-   Aggregated tasks:
    -   `test-latest = { depends-on = ["test312"] }`
    -   `test-all = { depends-on = ["test311", "test310", "test39"] }`

**Running Tests:**
-   To run tests for the latest supported Python (this is THE prefered way of development testing) version (currently 3.12):
    ```sh
    pixi run test-latest
    ```
-   To run tests for a specific Python version (e.g., Python 3.9):
    ```sh
    pixi run test39
    ```
    Alternatively, activate the environment and run pytest:
    ```sh
    pixi shell -e py39
    pytest -n auto
    ```
-   To run tests for all defined Python versions (excluding latest):
    ```sh
    pixi run test-all
    ```

Test configurations, such as paths and coverage, are defined in `[tool.pytest.ini_options]` and `[tool.coverage.run]` in [`pyproject.toml`](pyproject.toml).

## 4. CI/CD Pipeline Overview

The CI/CD pipeline is defined in [`.gitlab-ci.yml`](.gitlab-ci.yml). It automates testing, building, and deployment.

**Key CI Stages and Jobs:**

1.  **Pre-commit Checks (`test_topfarm_precommit`):**
    -   Uses `ghcr.io/prefix-dev/pixi:latest` image.
    -   Runs `pixi run -e default pre-commit run --all-files` to ensure code formatting and quality.

2.  **Testing on Linux:**
    -   **`test_topfarm`**:
        -   Runs on the `ghcr.io/prefix-dev/pixi:latest` image.
        -   Depends on `test_topfarm_precommit`.
        -   Executes `pixi run test-latest` (tests with the newest Python version).
        -   Uploads coverage reports (`./htmlcov`).
    -   **`test_topfarm_all_py_versions`** (manual trigger):
        -   Runs on the `ghcr.io/prefix-dev/pixi:latest` image.
        -   Executes `pixi run test-all` (tests with older Python versions).

3.  **Testing on Windows:**
    -   **`test_topfarm_windows`**:
        -   Runs on `registry.windenergy.dtu.dk/dockerimages/windows-pixi:ltsc2019` image.
        -   Depends on `test_topfarm_precommit`.
        -   Installs the package: `pixi run pip install .`
        -   Executes `pixi run test-latest`.
    -   **`test_topfarm_windows_all_py_versions`** (manual trigger):
        -   Runs on `registry.windenergy.dtu.dk/dockerimages/windows-pixi:ltsc2019` image.
        -   Installs the package: `pixi run pip install .`
        -   Executes `pixi run test-all`.

4.  **Documentation:**
    -   **`test_docs_build`**:
        -   Builds documentation using Sphinx: `pixi run -e default "cd docs; sphinx-build -j auto . build/html"`.
        -   Runs on merge requests to ensure docs build correctly.
    -   **`pages`**:
        -   Builds and deploys documentation to GitLab Pages.
        -   Uses the same Sphinx command as `test_docs_build`.
        -   Moves built HTML to `public/`.
        -   Runs on `master` branch and branches starting with `test_doc`.

5.  **Deployment:**
    -   **`test_topfarm_deploy`** (to TestPyPI):
        -   Runs on `master` branch.
        -   Builds the package: `pixi run -e default hatch build`.
        -   Publishes to TestPyPI: `pixi run -e default hatch publish -r test ...`.
    -   **`pypi_deploy`** (to PyPI):
        -   Runs on tags.
        -   Builds the package: `hatch build`.
        -   Publishes to PyPI: `hatch publish ...`.

## 5. Building Documentation

Documentation is built using Sphinx and nbsphinx for Jupyter notebook integration.
-   Configuration: [`docs/conf.py`](docs/conf.py)
-   Main page: [`docs/index.rst`](docs/index.rst)

To build the documentation locally:
1.  Ensure your `pixi` environment is active or use `pixi run`.
2.  Navigate to the `docs` directory.
3.  Run Sphinx:
    ```sh
    cd docs
    pixi run -e default sphinx-build -j auto . _build/html
    ```
    Or, if `sphinx-build` is in your pixi environment's PATH:
    ```sh
    pixi shell
    cd docs
    sphinx-build -j auto . _build/html
    ```
The output will be in `docs/_build/html`.

## 6. Coding Standards and Pre-commit

This project uses `pre-commit` to enforce coding standards and formatting. The configuration is in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).
To set up pre-commit hooks:
```sh
pixi run -e default pre-commit install
```
This will run checks automatically before each commit. To run all checks manually:
```sh
pixi run -e default pre-commit run --all-files
```

## 7. Versioning

Versioning is handled by `hatch-vcs`, which derives the version from Git tags. The version is written to [`topfarm/_version.py`](topfarm/_version.py) during the build process. This file is listed in [`.gitignore`](.gitignore).
Configuration is in [`pyproject.toml`](pyproject.toml):
```toml
[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "topfarm/_version.py"
```

## 8. Dependencies

Project dependencies are managed in [`pyproject.toml`](pyproject.toml):
-   Core dependencies: `[project.dependencies]`
-   Optional dependencies: `[project.optional-dependencies]` (e.g., `tensorflow`)
-   `pixi` specific dependencies (for development and CI tools): `[tool.pixi.dependencies]`
-   `pixi` Python version specific dependencies: `[tool.pixi.feature.python39.dependencies]`, etc.

To install with optional dependencies (e.g., tensorflow), refer to the installation guide in [`docs/installation.rst`](docs/installation.rst):
```sh
pip install topfarm[tensorflow]
```
Within a `pixi` managed environment, these optional dependencies would typically be included in the main `pixi.lock` if specified in the `pyproject.toml` features or directly.
