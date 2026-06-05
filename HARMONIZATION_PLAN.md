# Harmonization Plan for `km` / `validator` / `cox` / related repos

## Why this document exists

While migrating `strata-fit-v6-meta-algo-py` to standalone `run_context`, we hit repeated clean-environment failures caused by dependency drift between local workspace repos and the currently pinned published artifacts.

The core pattern was the same across multiple repos:

- the local workspace version had already moved toward standalone execution
- the pinned installable artifact still exposed wrapper-era package roots
- importing a submodule still executed package `__init__`
- package `__init__` pulled in `vantage6.algorithm` or other runtime-specific code
- clean builds failed even when the meta repo only needed pure helper modules

This document records the actions needed to harmonize the dependent repos, the preferred direction, and the success criteria.

## Current findings

### `strata-fit-v6-km-py`

- The clean installable artifact pinned by meta was still wrapper-era.
- Decision: if `strata-fit-v6-km-py` remains a standalone algorithm repo, it must move fully to `run_context`.
- The meta repo only needed KM preprocessing primitives, but importing `strata_fit_v6_km_py.preprocessing` still required a package root that imported legacy code.
- Result: meta clean-env validation could not rely on the published KM artifact.

### `v6-coxph-py`

- The package root imports wrapper-era `central` code.
- Decision: if `v6-coxph-py` remains a standalone algorithm repo, it must also move to `run_context`.
- Even `import coxph.contracts` triggers `coxph.__init__`, which imports `vantage6.algorithm`.
- Decision: we should not assume `v6-federated-core` stays forever. But it is not the first thing to remove. First clean the repo boundaries and eliminate wrapper-era package-root breakage. Then decide whether the remaining `v6-federated-core` features still justify a shared dependency.
- Result: meta clean-env validation could not rely on the published Cox artifact.

### `strata-fit-data-schema` / validator package

- The package root imports `v6_utils`, which imports `vantage6.algorithm.tools.decorators`.
- Decision: validator should remain a standalone repo. Near-term we need import-safe packaging and full `run_context` migration. Medium-term we should assess moving more schema generation and validation mechanics toward `datavalgen`, provided it does not break the existing validator algorithm contract.
- The meta repo only needed schema-driven validation logic, but importing `strata_fit_v6_data_validator_py.logic` still executed the package root.
- Decision: import minimization is a hard requirement. The validator package root must become safe, light, and free of runtime-only imports.
- Result: clean-env validation broke on import before any real validation logic ran.

### `strata-fit-v6-imputation-py`

- The external registry eager-imports all strategies at package import time.
- Decision: yes. Strategies should be imported lazily or explicitly, not eagerly at package import time.
- The mean strategy uses `polars`; mixed native/runtime environments caused a segfault in local validation.

- Decision: remove `polars` from the imputation package unless there is a demonstrated need that cannot be served by `pandas`/`numpy`.
- Result: even default mean-imputation paths were fragile if the external package was imported directly.

## Current execution status

This section should be updated as harmonization work lands so future sessions do not repeat completed migrations.

### Addressed

- `strata-fit-v6-imputation-py`
  - migrated to standalone `run_context`
  - Harbor wrapper removed
  - package root made import-safe
  - mean-imputation path no longer depends on `polars`
  - clean-env local verification passed
- `strata-fit-data-schema`
  - migrated to standalone `run_context`
  - Harbor wrapper removed
  - package root made import-safe
  - local clean-env tests and container smoke passed
- `v6-infrastructure-sh` skill
  - reusable infra guidance updated to include:
    - clean-env before infra
    - container smoke before infra
    - duplicate entrypoint trap
    - container-visible `RUN_CONTEXT_FILE` inputs
    - master-node traceback-first debugging
- `strata-fit-v6-meta-algo-py` signoff path
  - clean-env and local stress-matrix lanes no longer depend on published `km`/`cox` repos
  - infra baseline is now scripted as the default lane
  - full infra matrix remains available as an explicit expansion path

### Still open

- publish harmonized validator/imputation artifacts and update downstream pins
- decide whether `km` and `cox` survive as independent repos or get replaced by pure shared logic
- get the scripted infra baseline to green on amd64 CI and record any remaining harness-specific traps

## Agreed direction

This reflects the current preferred architecture.

1. Keep standalone `imputation` and `validator`.
   - They are reusable enough to justify dedicated repos.
   - Both must become import-safe, `run_context`-based, and deterministic to install.

2. Do not keep independent analysis repos in their current form.
   - `km` and `cox` should not remain half-library, half-algorithm dependencies that downstream repos install just to reach helper functions.

3. If independent survival logic is still needed, create one pure shared library.
   - Example: `strata-fit-v6-survival-core-py`
   - Put Cox math, KM preprocessing, types, and survival contracts there.
   - Keep standalone algorithm repos thin adapters or remove them entirely.

4. Re-evaluate `v6-federated-core` after repo boundaries are fixed.
   - Do not make it the first target.
   - First remove the obvious wrapper-era breakpoints.
   - Then decide whether the remaining shared functionality is still worth carrying as a dependency.

## Harmonization principles

These rules should apply to all shared STRATA-FIT algorithm repos.

1. Package roots must be import-safe.
   - `__init__.py` must not import wrapper entrypoints, decorators, or orchestration code.
   - `import package.submodule` must not fail because package root imported legacy runtime code first.

2. Pure logic must be separated from runtime wrappers.
   - Math, preprocessing, validation, and serialization helpers belong in pure modules.
   - `run_context` entrypoints should be thin adapters around pure functions.

3. No shared repo should require `vantage6-algorithm-tools` for import-time access to pure helpers.
   - If Vantage6 integration is still needed somewhere, it must live in an adapter layer, not the core package root.

4. Shared packages must be usable from clean builds with deterministic installs.
   - `pip install --no-deps` plus explicit runtime dependencies should work.
   - No package should rely on incidental transitive imports from unrelated repos.

5. Published artifacts must match the workspace contract.
   - If a local repo has migrated to standalone runtime, the published commit/tag consumed by downstream repos must also be migrated.

## Process traps to avoid next time

This section records the paths that burned time during the meta migration so a future session can skip them early.

### 1. Do not trust a published pin just because the local workspace repo looks fixed

- We repeatedly had a migrated local repo in `/media/share/strata-fit/...` while the published pin consumed by another repo still pointed at a wrapper-era artifact.
- Result: local manual inspection suggested the dependency was safe, but clean-env installs still failed immediately.
- Working rule:
  - verify the exact installed artifact, not just the workspace checkout
  - when downstream clean-env validation matters, prefer testing the published pin in a fresh venv before assuming the repo is harmonized

### 2. Do not start by removing `v6-federated-core`

- The first breakages were not caused by `v6-federated-core`; they were caused by package-root imports pulling runtime-only code.
- Time was wasted treating shared-core removal as the first cleanup target instead of fixing the import boundary first.
- Working rule:
  - first make package roots import-safe
  - second separate pure logic from runtime adapters
  - only then decide whether the remaining shared core is still justified

### 3. Treat package-root import safety as a gate before algorithm correctness

- Several repos failed before any real logic ran because `package.__init__` imported `central`, decorators, or helper modules that pulled `vantage6.algorithm`.
- This affected validator, Cox, KM, and earlier imputation work.
- Working rule:
  - validate these imports in a fresh environment before deeper debugging:
    - `import package`
    - `import package.logic_module`
    - `import package.helper_module`
  - if these fail, stop and fix package boundaries before testing behavior

### 4. Do not reuse a half-broken repo-local virtual environment

- We lost time on a repo-local `.venv` after an interrupted `pip` self-upgrade left pip partially broken.
- Symptoms were misleading import failures inside pip itself rather than the target package.
- Working rule:
  - use a fresh `/tmp/...` venv for verification runs
  - avoid debugging target-package issues inside an already suspect environment

### 5. Avoid large chained install-and-test commands during bootstrap

- Commands like `pip install ... && pip install ... && pytest ...` made it harder to see which step actually failed.
- When a dependency tree is unstable, opaque chained commands increase retry cost.
- Working rule:
  - run installation and import checks as separate commands
  - capture a `pip freeze` artifact once the environment is known-good

### 6. Standalone algorithm repos are poor helper-library dependencies unless proven otherwise

- Meta only needed pure helper modules from KM/Cox/validator/imputation, but those repos exposed mixed algorithm/library boundaries.
- This repeatedly forced vendoring or local compatibility shims.
- Working rule:
  - if a repo is consumed mainly for pure helpers, either:
    - extract a real shared pure library, or
    - stop consuming the algorithm repo as a dependency

### 7. Clean-env validation should use deterministic install order

- Resolver drift and native-extension surprises were easier to diagnose when installs happened in a fixed sequence.
- The most stable pattern was:
  1. create a fresh temp venv
  2. install stable primitives first
  3. install pinned git/tar dependencies with `--no-deps`
  4. install the target repo with `--no-deps`
  5. run import gates before full tests

### 8. Infra failures and package failures should be separated early

- Some failures looked like infrastructure issues but were actually package import/runtime breakages inside the algorithm container.
- Working rule:
  - first pass: prove local import and local runtime smoke in a clean venv
  - second pass: prove container `run_context` smoke
  - third pass: run `v6-infrastructure-sh`
  - when infra fails, inspect the master org node container for tracebacks before changing infra assumptions

### 9. Maintain a running migration manifest while work is in flight

- Once multiple repos diverge, context loss becomes expensive.
- Working rule:
  - record per-repo status, known traps, verification commands, and unresolved decisions in-repo while migration is ongoing
  - update this document whenever a repeated dead end is identified

### 10. Beware duplicate `run_context` entrypoints during editable-install smoke tests

- A runtime smoke run from the repo root after an editable install can see both:
  - the installed distribution metadata
  - the local source-tree `.egg-info`
- Result: `Duplicate run-context entrypoint '<name>'` even though the package is structurally fine.
- Working rule:
  - run dispatcher smokes from outside the repo root, or
  - remove stale local metadata before testing entrypoint resolution

### 11. Container smoke inputs must use container-visible paths

- A host path in `run_context.json` can pass local Python smoke and still fail in Docker with `FileNotFoundError`.
- Working rule:
  - for container/runtime smoke, mount the dataset beside the context file and use container-visible URIs only
  - do not reuse host-only absolute paths inside mounted run contexts

### 12. Reusable skills may be operationally correct but still miss migration knowledge

- The current `v6-infrastructure-sh` skill captured infra lifecycle steps correctly, but it did not capture the repo-boundary and clean-env failure patterns that dominated recent work.
- In this environment the skill file was root-owned, so the guidance could not be persisted directly without elevated credentials.
- Working rule:
  - review reusable skills before relying on them as complete runbooks
  - if a skill cannot be updated immediately, record the missing guidance here so it is not lost

### 13. Default `INFRA_DIR` assumptions may be wrong in this workspace layout

- `tests/infra/run_local_infra_smoke.sh` defaults to `../v6-infrastructure-sh` relative to the meta repo.
- In this workspace the real harness path is `/media/share/v6-infrastructure-sh`, so the default path fails immediately.
- Working rule:
  - treat `INFRA_DIR` as required when the workspace layout differs from the expected sibling checkout
  - record the actual local harness path in the active manifest before first infra runs

### 14. Never hand `infra.sh` a shared project virtualenv

- The harness upgrades `pip`, `setuptools`, and `wheel` inside the environment it receives via `PYTHON_INTERPRETER` / `VENV_PATH`.
- Using the repo’s working `.venv` risks cross-run mutation, version downgrades, and misleading follow-up failures.
- Working rule:
  - create a disposable `/tmp/...` venv specifically for infra runs
  - allow the harness to mutate only that throwaway environment

### 15. Infra evaluation may be blocked by external Harbor reachability, not by algorithm code

- The meta baseline infra attempt reached `pull_docker_images()` and then failed on:
  - `harbor2.vantage6.ai/infrastructure/server:4.13.3`
  - `dial tcp 20.50.179.220:443: i/o timeout`
- Direct `curl` and `docker pull` checks to Harbor timed out from the host as well.
- The public `vantage6.ai` incident notice states:
  - Harbor was shut down after unauthorized registry access on April 2, 2026
  - on May 11, 2026 Vantage6 4.15 moved infra images to GitHub Container Registry
  - Harbor itself is being discontinued
- Working rule:
  - do not attempt to revive Harbor-based defaults
  - switch infra tests to GHCR-backed `server-lite` / `node-lite` / `ui` images or a local mirror of them

### 16. Keep node image selection explicit in node YAML, not only in shell env

- For real node operations, hidden image defaults in shell scripts are too opaque.
- The effective node runtime should remain visible in the node config alongside:
  - `policies.allowed_algorithms`
  - `run_context_file: true`
  - `share_config: false`
  - `share_algorithm_logs: false`
  - `prometheus.enabled: false`
  - database `mount_mode`
- Working rule:
  - even if the harness injects image names via env vars, production-like node configs should explicitly include `images.node`
  - prefer digest-pinned image refs in operator-facing configs so reviews can verify both allowed algorithms and the node runtime in one place
  - do not keep changing algorithm code when the failure is an unreachable or retired server/node image registry

### 16. GHCR replacement infra images may be amd64-only

- The `server-lite` / `node-lite` tags used to replace Harbor were not available as native `linux/arm64` manifests on this host.
- Pulling them without a platform override failed before infra boot.
- Working rule:
  - mirror GHCR infra images with an explicit source platform, e.g. `--platform linux/amd64`
  - assume emulation may be required on ARM developer machines until native tags are confirmed

## How to use this document during future migrations

Before migrating another repo:

1. Check whether the repo is meant to stay standalone or become pure shared logic.
2. Test package-root import safety in a fresh temp venv.
3. Verify the published artifact, not only the local workspace tree.
4. Prefer deterministic installs with `--no-deps` for conflict-prone git/tar dependencies.
5. Only move to infra after local clean-env and container smoke are green.

## Repo-by-repo actions

### `strata-fit-v6-km-py`

Required actions:

- Publish a standalone release that matches the migrated local workspace version.
- Ensure `strata_fit_v6_km_py.__init__` is lazy or import-safe.
- Keep preprocessing and types modules free of wrapper/runtime imports.
- Add clean-env tests that verify:
  - `import strata_fit_v6_km_py`
  - `import strata_fit_v6_km_py.preprocessing`
  - `import strata_fit_v6_km_py.types`
  all succeed without `vantage6.algorithm`.

Preferred structure:

- `preprocessing.py`
- `types.py`
- `runtime.py`
- `container.py`
- `partial.py`
- `central.py`

Success criteria:

- clean-env import of root and helper modules passes
- Docker build uses `python:3.11-slim`
- `RUN_CONTEXT_FILE` smoke passes
- downstream repos can consume a published pin without vendoring KM code

### `v6-coxph-py`

Required actions:

- Split pure Cox logic from any Vantage6-specific orchestration.
- Make `contracts.py` and `methods.py` importable without package-root side effects.
- Remove `vantage6.algorithm` imports from package root.
- Publish a standalone/import-safe release.

Preferred structure:

- `contracts.py`
- `methods.py`
- optional `runtime.py` / `container.py` if the repo is meant to run as a standalone algorithm
- no wrapper imports from `__init__.py`

Success criteria:

- `import coxph`
- `import coxph.contracts`
- `import coxph.methods`
  all succeed in a clean venv without `vantage6.algorithm-tools`
- downstream repos no longer need to vendor Cox helper code

### `strata-fit-data-schema` / validator package

Required actions:

- Separate schema/config/validation logic from Vantage6 helper functions.
- Move `validate_data` or any decorator-based utilities into an adapter module that is not imported by package root.
- Keep `logic.py`, `schema.py`, and config access import-safe.

Preferred structure:

- package root exports schema/logic only
- `v6_utils.py` remains optional and isolated
- `__init__.py` must not import `v6_utils.py`

Success criteria:

- `import strata_fit_v6_data_validator_py`
- `import strata_fit_v6_data_validator_py.logic`
- `import strata_fit_v6_data_validator_py.schema`
  all succeed without `vantage6.algorithm`
- config bootstrap remains deterministic in containers and clean envs

### `strata-fit-v6-imputation-py`

Required actions:

- Stop eager-importing all strategies at package import time.
- Avoid `polars` in the default/simple strategy path unless strictly necessary.
- Provide a lazy registry or explicit strategy imports.
- Add clean-env and import-only tests for mean and MICE paths.

Preferred structure:

- `base.py` defines enum + registry API only
- strategy modules register lazily or through explicit import points
- no native-heavy import side effects in package root

Success criteria:

- default mean-imputation import path does not load `polars`
- importing the package does not segfault in mixed environments
- downstream repos can choose simple strategies without native runtime surprises

## Do we want independent `cox` / `km` repos at all?

This needs an explicit decision. Right now the independent repos behave partly like shared libraries and partly like standalone algorithms. That ambiguity is the source of most of the breakage.

### Option A: Keep independent `cox` and `km` repos

Choose this only if both are true:

- they are consumed by multiple repositories or teams as shared reusable packages
- we are willing to maintain release discipline and clean-env CI for them as first-class libraries

If we keep them:

- they must become pure-core-first packages
- standalone/runtime adapters must be thin and optional
- each repo needs its own clean-env CI gate and published versioning discipline

### Option B: Do not keep independent algorithm-style repos

Choose this if:

- `cox` and `km` are mostly implementation details of meta and closely related survival algorithms
- there are not multiple external consumers that genuinely need them as separate packages

If we do not keep them:

- move pure shared logic into one focused library, for example `strata-fit-v6-survival-core-py`
- keep algorithm repos thin and self-contained
- avoid depending on partially maintained standalone algorithm repos just for helper functions

### Recommendation

Keep standalone `imputation` and `validator`.

Do not keep independent `cox` and `km` repos in their current mixed form.

Better choices are:

1. If there are multiple consumers:
   - create a pure shared library such as `strata-fit-v6-survival-core-py`
   - move Cox math, KM preprocessing, types, and shared contracts there
2. If there are not multiple consumers:
   - keep the logic inside the consuming algorithm repos and stop publishing separate `cox` / `km` packages

The current model, where downstream repos install standalone-algorithm repos to access helper functions, is too fragile.

## Recommended sequencing

1. Stabilize the two standalone repos we want to keep.
   - imputation first
   - validator second

2. Add import-safety CI to each kept shared repo.
   - clean venv
   - `pip install --no-deps` where appropriate
   - package root import
   - helper-module import

3. Stop treating `cox` and `km` as downstream install-time dependencies.
   - choose between survival-core extraction or inlining into consumers
   - update meta and any other consumers accordingly

4. Revisit `v6-federated-core`.
   - keep it only if the remaining shared abstractions still pay for themselves
   - otherwise move the needed pieces local or into more focused shared packages

5. Remove vendored stopgaps from meta once upstream artifacts are fixed or intentionally replaced.
   - local KM preprocessing copy
   - local Cox helper copy
   - local validator logic copy
   - local imputation registry if upstream becomes import-safe

## Execution plan

### Phase 1: Imputation repo

Target repo:

- `strata-fit-v6-imputation-py`

Actions:

- remove `polars`
- change eager strategy registration to lazy or explicit registration
- make package root import-safe
- add `run_context` runtime/container surface
- add clean-env install/import tests

Success criteria:

- `import strata_fit_v6_imputation_py` succeeds in a clean env
- mean-imputation path does not import `polars`
- clean-env tests pass with `--no-deps` driven install order
- package can be consumed by meta without local compatibility shims

### Phase 2: Validator repo

Target repo:

- `strata-fit-data-schema`

Actions:

- isolate `v6_utils.py` from package root
- keep schema/config/logic import-safe
- migrate standalone algorithm execution to `run_context`
- assess where `datavalgen` can replace internal schema generation without breaking the public validator behavior
- add clean-env import tests for root, `logic`, and `schema`

Success criteria:

- package root and logic/schema imports work without `vantage6.algorithm`
- validator algorithm runs through `RUN_CONTEXT_FILE`
- meta no longer needs local validator compatibility code

### Phase 3: Survival analysis boundary

Target repos:

- `strata-fit-v6-km-py`
- `v6-coxph-py`
- possibly new `strata-fit-v6-survival-core-py`

Actions:

- decide whether to extract a shared survival-core repo or inline survival logic into consumers
- if keeping standalone algorithm repos, move them to `run_context`
- stop depending on algorithm-package roots for helper imports

Success criteria:

- meta no longer installs standalone `km` / `cox` repos as helper dependencies
- shared survival logic, if any, lives in a pure import-safe package
- all dependent repos pass clean-env validation against published pins

## Success criteria for the overall harmonization effort

Technical success:

- every repo builds from `python:3.11-slim`
- every repo supports `RUN_CONTEXT_FILE`
- no shared helper import requires `vantage6-algorithm-tools`
- no package root import triggers wrapper-only code
- `pip install --no-deps` based deterministic installs succeed where intended
- clean-env validation passes for meta without vendored emergency copies

Integration success:

- meta clean-env lane passes from published dependency pins
- local stress matrix passes
- infra lane passes for:
  - `baseline_3n_full`
  - `fanout_5n_survival`
  - `fanout_8n_km`
  - `fanout_8n_cox`

Operational success:

- downstream repos do not need to discover import-time breakage by accident
- the published package contract matches the workspace contract
- release notes clearly state when a repo is safe for standalone downstream consumption

## Practical definition of done

This harmonization is done when:

- meta can remove its local vendored compatibility layers
- meta installs cleanly against published pinned artifacts only
- all repo-local and infra validation lanes pass
- the team has made a clear keep-vs-consolidate decision for independent Cox/KM repositories
