#!/usr/bin/env bash
# Build flash-next-swarm-quad3090 v1.0: halt95's Merlin engine v2.5.2 (built by halt95's own scripts/build-v2.5.sh,
# unchanged) plus this fork's 40 commits, one rebuilt compiled extension and FlashInfer 0.7.
#
#   scripts/build-swarm.sh [SRC_DIR] [VENV_DIR]          (defaults: ./vllm-swarm ./venv-swarm)
#
# Inputs:
#   halt95's v2.5.2 release assets, as scripts/build-v2.5.sh documents them: BUNDLE (v2.5.2-from-upstream-v0.30.0.bundle,
#     .gz accepted) and ARTIFACTS (build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz), from
#     https://github.com/halt95/merlin-engine/releases/tag/v2.5.2 and .../v2.2.0. Both are hash-checked against
#     upstream/PIN-v2.5.
#   MOE_C_TAR  this fork's rebuilt MoE extension (release asset, upstream/PIN-swarm-v1.0 moe_C_tar); downloaded from the
#              fork's release when not given. Needs glibc >= 2.38.
#   BUILD_OWN=1  build both extensions from this fork's tree instead (cmake >= 3.26, ninja, a C/C++ compiler,
#              nvcc >= 13.0; 15-30 min; your hashes will differ from the pin's). halt95's step still uses its ARTIFACTS
#              tarball for the unchanged build products (stock vLLM 0.30.0 wheel files, glibc >= 2.28).
#
# Steps:
# 1. scripts/build-v2.5.sh SRC VENV: the v2.5.2 tree at halt95's commit, the pinned venv and halt95's build products,
#    all verified by halt95's script. Skipped when SRC already holds this fork's build (marker .swarm-build).
# 2. Fetch this fork's bundle into SRC (prerequisite: the v2.5.2 commit), check out, assert commit AND tree against
#    upstream/PIN-swarm-v1.0.
# 3. The MoE extension (tarball, hash-checked) or BUILD_OWN=1 (both extensions from the tree).
# 4. FlashInfer 0.7.0.post1 and nccl-extensions into the venv (--require-hashes --no-deps), then pip check.
# 5. Re-hash every build product (the same list and format as build-v2.5.sh), so serve-swarm.sh's integrity check
#    covers the new extension too.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
sha_ok(){ [ "$(sha256sum "$2" | cut -d" " -f1)" = "$1" ]; }
abspath(){ case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
SRC="$(abspath "${1:-vllm-swarm}")"; VENV="$(abspath "${2:-venv-swarm}")"
# shellcheck disable=SC1091
. "$HERE/upstream/PIN-swarm-v1.0"
REL25="$HERE/release/v2.5"; RELS="$HERE/release/swarm-v1.0"
SWARM_MARK="$SRC/.swarm-build"
SUMS="$SRC/.v2.5-build-products.sha256"
write_sums(){
  (cd "$SRC" && while read -r p; do [ -n "$p" ] || continue; find "${p%/}" -type f; done < "$REL25/build-artifacts.list" \
    | grep -vx 'vllm/_version.py' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$SUMS.tmp" && mv "$SUMS.tmp" "$SUMS"; }

# glibc check for the prebuilt extension, before anything is fetched
if [ "${BUILD_OWN:-0}" != 1 ]; then
  have_glibc="$(ldd --version 2>/dev/null | sed -n '1s/.* \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p')"
  if [ -n "$have_glibc" ] && [ "$(printf '%s\n%s\n' "$min_glibc" "$have_glibc" | sort -V | head -1)" != "$min_glibc" ]; then
    echo "the prebuilt MoE extension needs glibc >= $min_glibc (this system: $have_glibc); use BUILD_OWN=1"; exit 1
  fi
fi

# 1. halt95's v2.5.2 build
if [ -f "$SWARM_MARK" ]; then
  echo "step 1: $SRC already holds this fork's build; halt95's build step is not re-run"
else
  # halt95's tarball route always (it supplies the unchanged build products); BUILD_OWN applies to step 3 only, so
  # the two extensions are compiled once, from this fork's tree
  env -u BUILD_OWN "$HERE/scripts/build-v2.5.sh" "$SRC" "$VENV"
fi

# 2. this fork's source
sha_ok "$bundle_sha256" "$HERE/$bundle" || { echo "bundle sha256 mismatch: $HERE/$bundle"; exit 1; }
if ! git -C "$SRC" rev-parse -q --verify "$commit^{commit}" >/dev/null 2>&1; then
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$base_commit" ] || { echo "$SRC is not at the v2.5.2 commit $base_commit"; exit 1; }
  [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has modified tracked files"; exit 1; }
  git -C "$SRC" fetch -q "$HERE/$bundle" "+$ref:refs/heads/$tag"
fi
[ "$(git -C "$SRC" rev-parse -q --verify HEAD)" = "$commit" ] || git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$commit"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$commit" ] || { echo "checkout failed"; exit 1; }
got_tree="$(git -C "$SRC" rev-parse 'HEAD^{tree}')"
[ "$got_tree" = "$tree" ] || { echo "tree $got_tree != $tree"; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has modified tracked files"; exit 1; }
echo "source: $tag = $commit, tree $tree verified (on halt95's v2.5.2 = $base_commit)"

# 3. compiled extension(s); a re-run over a finished build only re-verifies them
prev_route=""; [ -f "$SWARM_MARK" ] && prev_route="$(awk '{print $3}' "$SWARM_MARK")"
want_route=tarball; [ "${BUILD_OWN:-0}" = 1 ] && want_route=own
if [ "$prev_route" = "$want_route" ] && (cd "$SRC" && sha256sum -c --quiet --strict "$SUMS" >/dev/null 2>&1) \
   && { [ "$want_route" = own ] || sha_ok "$moe_C_sha256" "$SRC/$moe_C"; }; then
  route=$prev_route; echo "compiled extensions: already installed ($route), all build product hashes re-verified"
elif [ "${BUILD_OWN:-0}" = 1 ]; then
  NVCC="${NVCC:-$(ls -d "$VENV"/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc 2>/dev/null | head -1)}"
  [ -x "${NVCC:-}" ] || { echo "BUILD_OWN=1 needs nvcc 13.0 or newer: set NVCC=/path/to/nvcc"; exit 1; }
  BD="$SRC/.build-swarm"; CUDA_BIN="$(dirname "$NVCC")"
  env PATH="$CUDA_BIN:$PATH" TORCH_CUDA_ARCH_LIST="$build_arch" MAX_JOBS="${MAX_JOBS:-8}" \
    cmake -S "$SRC" -B "$BD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_CUDA_COMPILER="$NVCC" > "$BD.configure.log" 2>&1 \
    || { echo "cmake configure failed; see $BD.configure.log"; exit 1; }
  for tgt in _C_stable_libtorch _moe_C_stable_libtorch; do
    env PATH="$CUDA_BIN:$PATH" TORCH_CUDA_ARCH_LIST="$build_arch" \
      cmake --build "$BD" --target "$tgt" -j "${MAX_JOBS:-8}" > "$BD.$tgt.log" 2>&1 || { echo "build $tgt failed; see $BD.$tgt.log"; exit 1; }
    so="$(find "$BD" -name "$tgt.abi3.so" | head -1)"; cp "$so" "$SRC/vllm/$tgt.abi3.so"
    echo "built $tgt sha256=$(sha256sum "$SRC/vllm/$tgt.abi3.so" | cut -d' ' -f1)"
  done
  route=own
else
  MOE_C_TAR="$(abspath "${MOE_C_TAR:-$HERE/$moe_C_tar}")"
  if [ ! -f "$MOE_C_TAR" ]; then
    curl -fsSL -o "$MOE_C_TAR.part" "$moe_C_url" && mv "$MOE_C_TAR.part" "$MOE_C_TAR" \
      || { rm -f "$MOE_C_TAR.part"; echo "could not fetch $moe_C_url; download it and set MOE_C_TAR="; exit 1; }
  fi
  sha_ok "$moe_C_tar_sha256" "$MOE_C_TAR" || { echo "MoE extension tarball sha256 mismatch: $MOE_C_TAR"; exit 1; }
  tar -xzf "$MOE_C_TAR" -C "$SRC" "$moe_C"
  sha_ok "$moe_C_sha256" "$SRC/$moe_C" || { echo "extracted $moe_C does not match the pin"; exit 1; }
  route=tarball
fi

# 4. FlashInfer 0.7
"$VENV/bin/python" -m pip install -q --require-hashes --no-deps -r "$RELS/requirements-swarm.txt"
"$VENV/bin/python" -m pip check >/dev/null || { "$VENV/bin/python" -m pip check; exit 1; }
got_fi="$("$VENV/bin/python" -c 'import flashinfer; print(flashinfer.__version__)')"
[ "$got_fi" = "$flashinfer" ] || { echo "flashinfer $got_fi, expected $flashinfer"; exit 1; }

# 5. re-hash the build products
write_sums
echo "$route $(sha256sum "$SRC/vllm/_C_stable_libtorch.abi3.so" | cut -d' ' -f1) $(sha256sum "$SRC/$moe_C" | cut -d' ' -f1)" > "$SRC/.v2.5-build-products"
echo "$tag $commit $route" > "$SWARM_MARK"
(cd / && PYTHONPATH="$SRC" "$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)")
echo "done: TREE=$SRC VENV=$VENV; next: config/swarm.env (scripts/size-ram-tier.sh), then scripts/run-swarm.sh"
