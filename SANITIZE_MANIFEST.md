# Sanitize manifest: mamba3 (2026-10-31)

Branch `chore/sanitize`, cut from `master` @ `12c168f`. 6 local commits, including this file. Nothing pushed. PR #3 (@d7aa9ec) is included as the first commit.

## Removed
| Path | Why |
|---|---|
| `AGENTS.md`, `mamba3/AGENTS.md` | Operator notes for an AI agent. |

Local paths were scrubbed:
- `file:///home/...` link in REFERENCE_THIS_SHIT.md:165;
- personal mount points in TRAINING_UBUNTU.md:13-15 (now `DATASET_DIR` / `DATASET_ROOT`);
- default dataset paths in extract_f0.sh:15 and run_train.sh:17 (now `./datasets/musdb18hq`).

## Moved
- `mamba3/REFERENCE_THIS_SHIT.md` → `docs/edge-controller-reference.md` (status updated to 2026-10-31).
- `mamba3/TRAINING.md`, `TRAINING_UBUNTU.md`, `TRAINING_WINDOWS.md` and `runpod_guide.md` → `docs/training/` (renamed `bazzite-distrobox-rocm.md`, `ubuntu-rocm.md`, `windows-rocm.md`, `runpod.md`).
- `edge_dsp_meta_prompt/specs/*.md` (6 files) → `docs/specs/`, with a README saying they're concept docs.
- References updated in code comments, scripts and requirements.txt.

## Added
`scripts/ci.sh` (mirrors the python-cpu job; `--rust` adds the cargo steps), `docs/specs/README.md`, and `Cargo.lock` (so the rc versions stop drifting). LICENSE is Apache-2.0 and unchanged. Scott's MIT rule is only for repos with no licence.

## Code fixes (tested)
- **`plugin/Cargo.toml`:**
  - The nih_plug feature `assert_no_alloc` doesn't exist at the pinned rev; it's `assert_process_allocs`.
  - `default-features = false` drops vst3: vst3-sys is GPLv3, and only `nih_export_clap!` is used.
- **`ort`:** exact pin `=2.0.0-rc.13`, `default-features = false` (+std, load-dynamic, ndarray), so openssl-sys isn't a build dependency.
- **`core/src/observer.rs`:** f_min/f_max typed f32 (E0689 on `.ln()`).

## Doc claims changed
- The title no longer claims Mamba-3 MIMO parity. It now says the code follows the paper's MIMO step and isn't parity-tested.
- The plugin is described as not compiling (26 errors) instead of "embeds it in a Rust CLAP plugin".
- Gate results are given with machine and versions. CI is described as blocked on Actions billing.
- Added a MUSDB18-HQ non-commercial note and the ONNX sha256.

## Tests, before and after (shared 8-vCPU Intel Xeon VM, Linux, Rust 1.97.1, Python with torch on CPU)
| Check | Before (`master` @ 12c168f) | After |
|---|---|---|
| Python gates (smoke_test, test_smoke, test_controller 9/9, ruff) | pass | pass (`scripts/ci.sh`) |
| `cargo check --workspace` | failed at dependency resolution (nonexistent nih_plug feature) | resolves; core compiles; `edge_dsp_plugin` stops at 26 errors (API mismatch with the pinned nih-plug/ort) |
| `cargo test -p edge_dsp_core` | did not build | 8 passed, 3 failed (`test_svf_lowpass`, `test_adaa_clipper`, `loud_signal_is_attenuated`) |
Logs: `_logs/mamba3-before-python.log`, `mamba3-before-ruff.log`, `mamba3-after-ci.log`, `mamba3-after-cargo-check.log`, `mamba3-after-core-test.log`.

The 26 plugin errors and 3 DSP test failures are real bugs. They weren't trivial, so they're left for the fix plan (M3, M4) and not hidden.

## Needs Scott
- **Plugin port:** porting the plugin crate to the pinned nih-plug/ort APIs is more than a trivial fix.
- **CI billing.**
- **History rewrite approval** (below).

## History rewrite (prepared, not run on the real repo)

Nothing has been rewritten or pushed. The commands below were tried on a throwaway `--mirror` clone in `/tmp` on 2026-10-31. They exited 0, and afterwards `git log --all -p` and the commit messages had 0 matches for the local-path patterns. Pack size went from 894 KiB to about the same. That scratch clone was deleted.

The secret scans (gitleaks `git` and `dir`, and trufflehog, over all history) found no secrets. So this rewrite is about local paths and large or unneeded files, not live keys. Scott says any exposed keys were rotated anyway.

`paths-mamba3.txt` (paths removed from every commit):
```text
AGENTS.md
mamba3/AGENTS.md
```

`replace.txt` is kept outside the repo, so the paths it removes aren't republished here. Use `/workspace/handoff/_history/replace-paths.txt`. It has 8 literal lines covering the Linux, Fedora Atomic (`/var/home`), macOS and Windows home-directory prefixes and the `x:` workspace URI. The `/var/home` line comes before the plain `/home` line.

Commands:
```bash
# 1. Merge chore/sanitize into the default branch first, so the rewrite covers the cleaned tree.
git clone --mirror https://github.com/sgm-audio/mamba3.git mamba3-rewrite.git
cd mamba3-rewrite.git
git filter-repo --invert-paths --paths-from-file ../paths-mamba3.txt --replace-text ../replace.txt
# 2. Check before pushing
git log --all -p | grep -cEf ../path-patterns.txt   # expect 0 (patterns file is next to replace.txt)
gitleaks git . --no-banner --redact                                                       # expect no leaks
git count-objects -vH
# 3. Push. Needs Scott's OK: this rewrites every commit hash.
git push --force --all origin && git push --force --tags origin
```

Caveats:
- GitHub keeps `refs/pull/*` for every PR, and a force-push can't touch them. The old objects stay reachable through those PR refs and cached commit URLs until GitHub Support runs a GC; ask them to purge after the push.
- Forks and existing clones keep the old history. Collaborators need to re-clone, and open PRs have to be recreated.
- `mamba3/edge_dsp_meta_prompt/mamba3_controller_quant.onnx` is still used (the README gives its sha256), so it stays.
