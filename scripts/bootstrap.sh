#!/usr/bin/env bash
#
# Bring a Lambda instance to a working state — with or without the volume.
#
#   bash bootstrap.sh            # full bootstrap
#   bash bootstrap.sh --backup   # just save Claude state
#
# TWO MODES, auto-detected by whether the persistent volume is mounted:
#
# VOLUME MODE (the original home box). Lambda wipes the root disk (/dev/vda1)
# on instance termination; only the virtiofs volume at
# /lambda/nfs/farhan-algoverse-summer26 survives. The repo, its .venv, .env and
# the HuggingFace weight cache already live there, so this does NOT reinstall
# Python packages or re-download models (~62GB saved).
#
# PORTABLE MODE (a bigger-GPU instance, where Lambda will not attach the
# volume). Nothing is on disk, so this script builds it: clones the repo,
# creates the venv, installs pinned deps, and points HF_HOME at local disk.
# Everything it needs is self-contained in THIS FILE, because it travels alone
# (scp) to a box that has never seen the volume.
#
#   scp bootstrap.sh ubuntu@<big-gpu-ip>:~/
#   ssh ubuntu@<big-gpu-ip> 'bash ~/bootstrap.sh'
#   ssh ubuntu@<big-gpu-ip> 'nano ~/Algoverse-Bias-Steering/.env'   # paste a key
#
# Override with env vars: BASE (install root, default $HOME), BRANCH (default
# exp/anchors — the only branch with runnable 7B configs + the snapshot loader),
# REPO_URL.
#
# THE OPENAI KEY IS NEVER ACCEPTED BY THIS SCRIPT — not via env var, not via a
# copied file. It creates .env with an empty OPENAI_API_KEY= and tells you to
# paste a fresh one in. A key on an ssh command line leaks into shell history and
# `ps`; a copied .env leaves live credentials on every disk it touches; and a
# .env.bak sibling is NOT covered by .gitignore (which lists only .env/.envrc),
# so it is committable. Paste, then rotate the old key.
#
# Model weights are not in git either and will re-download (~15GB for a 7B).
#
# Claude state is COPIED, never moved or symlinked. Both directions use
# `rsync -a --update` with no --delete, so nothing is ever destroyed and the
# newer copy of any file wins. That means:
#   - it is safe to run with a Claude session live (transcripts are append-only
#     JSONL; a partial copy self-heals on the next backup), and
#   - Claude keeps working normally if the volume is ever missing.
# The tradeoff vs a symlink: new sessions only reach the volume when a backup
# runs. Cron it — see the hint printed at the end.
#
# Safe to re-run: every step is idempotent.

set -uo pipefail

VOL="/lambda/nfs/farhan-algoverse-summer26"
REPO_URL="${REPO_URL:-https://github.com/Darksharkthe1st/Algoverse-Bias-Steering.git}"
# exp/anchors carries the snapshot dataset loader, datasets/Snapshots/, and the
# configs/exp/*.py campaign files. main has none of them, so a default clone of
# main cannot run any of these experiments.
BRANCH="${BRANCH:-exp/anchors}"

# PORTABLE is the whole switch: 1 = no volume, build everything from scratch.
if mountpoint -q "$VOL" 2>/dev/null; then
    PORTABLE=0
    BASE="$VOL"
else
    PORTABLE=1
    BASE="${BASE:-$HOME}"
fi

REPO="$BASE/Algoverse-Bias-Steering"
OPS="$BASE/_ops"
CLAUDE_STORE="$BASE/.claude-home"
CLAUDE_JSON_STORE="$BASE/.claude-home.json"
HF_CACHE="$BASE/.hf_cache"

# Alert when the volume exceeds this (GB). Override: THRESHOLD_GB=300 bash bootstrap.sh
THRESHOLD_GB="${THRESHOLD_GB:-200}"

MODE="${1:-full}"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

mkdir -p "$OPS"

# --------------------------------------------------------------------------
# Claude state sync (copy both ways, newest wins, nothing deleted)
# --------------------------------------------------------------------------
backup_claude() {
    [ -d "$HOME/.claude" ] || { warn "no ~/.claude to back up"; return; }
    mkdir -p "$CLAUDE_STORE"
    rsync -a --update "$HOME/.claude/" "$CLAUDE_STORE/" 2>/dev/null \
        && ok "backed up ~/.claude -> volume ($(du -sh "$CLAUDE_STORE" 2>/dev/null | cut -f1))" \
        || bad "backup FAILED"
    [ -f "$HOME/.claude.json" ] && cp -pu "$HOME/.claude.json" "$CLAUDE_JSON_STORE" 2>/dev/null \
        && ok "backed up ~/.claude.json"
    if [ -d "$CLAUDE_STORE/projects" ]; then
        ok "$(find "$CLAUDE_STORE/projects" -name '*.jsonl' 2>/dev/null | wc -l) session transcript(s) on volume"
    fi
}

restore_claude() {
    if [ ! -d "$CLAUDE_STORE" ]; then
        warn "no saved Claude state on the volume yet (first run — will be created by backup)"
        return
    fi
    mkdir -p "$HOME/.claude"
    rsync -a --update "$CLAUDE_STORE/" "$HOME/.claude/" 2>/dev/null \
        && ok "restored volume -> ~/.claude ($(find "$HOME/.claude/projects" -name '*.jsonl' 2>/dev/null | wc -l) transcripts)" \
        || bad "restore FAILED"
    [ -f "$CLAUDE_JSON_STORE" ] && cp -pu "$CLAUDE_JSON_STORE" "$HOME/.claude.json" 2>/dev/null \
        && ok "restored ~/.claude.json"
}

if [ "$MODE" = "--backup" ]; then
    printf '\n\033[1mClaude state backup\033[0m — %s\n\n' "$(date -u '+%Y-%m-%d %H:%M UTC')"
    backup_claude
    echo
    exit 0
fi

printf '\n\033[1mLambda instance bootstrap\033[0m — %s\n' "$(date -u '+%Y-%m-%d %H:%M UTC')"

# --------------------------------------------------------------------------
step "0. Persistent volume"
if [ "$PORTABLE" = 0 ]; then
    ok "volume mounted ($(df -h "$VOL" | awk 'NR==2{print $3}') used)"
    ok "VOLUME MODE — repo, venv, .env and weights are already here"
else
    warn "$VOL is not a mountpoint — running in PORTABLE MODE."
    warn "This is expected on a bigger-GPU instance, where Lambda will not"
    warn "attach the volume. Everything gets built under: $BASE"
    warn "Root disk is EPHEMERAL — copy results off before terminating:"
    warn "  scp -r ubuntu@<this-ip>:$REPO/runs ."
fi

# --------------------------------------------------------------------------
step "1. System packages"
APT_WANTED=()
for pkg in gh rsync tmux jq curl git python3-venv build-essential; do
    dpkg -s "$pkg" >/dev/null 2>&1 || APT_WANTED+=("$pkg")
done
if [ ${#APT_WANTED[@]} -gt 0 ]; then
    warn "installing: ${APT_WANTED[*]}"
    sudo apt-get update -qq >/dev/null 2>&1
    # gh ships in Ubuntu 22.04's universe repo; if that ever fails, fall back to
    # GitHub's own apt repo rather than leaving the user without it.
    if ! sudo apt-get install -y -qq "${APT_WANTED[@]}" >/dev/null 2>&1; then
        warn "bulk install failed; retrying individually"
        for pkg in "${APT_WANTED[@]}"; do
            sudo apt-get install -y -qq "$pkg" >/dev/null 2>&1 || bad "could not install $pkg"
        done
    fi
fi
for pkg in gh rsync tmux jq curl git; do
    command -v "$pkg" >/dev/null 2>&1 && ok "$pkg" || bad "$pkg MISSING"
done

# --------------------------------------------------------------------------
step "2. Claude Code (no Node required)"
# Claude Code ships as a self-contained native binary. npm is only ever a
# delivery vehicle for it — per the docs, "the installed claude binary does not
# itself invoke Node", and `file $(readlink -f $(which claude))` on an npm
# install reports an ELF executable, not a JS shim. So we use the native
# installer and keep Node off the dependency list entirely.
#
# It installs to ~/.local/bin/claude, which Ubuntu's default ~/.profile already
# puts on PATH, and it auto-updates in the background (npm installs do not).
export PATH="$HOME/.local/bin:$PATH"

install_claude_apt() {
    # Preferred: signed apt repo. More robust than `curl | bash` because the key
    # is fingerprint-verified before use, so a proxy/Cloudflare page returning
    # HTML fails loudly at the gpg check instead of being piped into a shell.
    local KEY=/etc/apt/keyrings/claude-code.asc
    local FPR=31DDDE24DDFAB679F42D7BD2BAA929FF1A7ECACE
    sudo install -d -m 0755 /etc/apt/keyrings || return 1
    sudo curl -fsSL https://downloads.claude.ai/keys/claude-code.asc -o "$KEY" || return 1
    # Verify the key really is Anthropic's before trusting the repo.
    gpg --show-keys --with-colons "$KEY" 2>/dev/null | grep -q "$FPR" || {
        bad "signing key fingerprint MISMATCH — refusing to add the repo"
        sudo rm -f "$KEY"; return 1
    }
    echo "deb [signed-by=$KEY] https://downloads.claude.ai/claude-code/apt/stable stable main" \
        | sudo tee /etc/apt/sources.list.d/claude-code.list >/dev/null || return 1
    sudo apt-get update -qq >/dev/null 2>&1
    sudo apt-get install -y -qq claude-code >/dev/null 2>&1
}

install_claude_native() {
    # Fallback: the native installer. Fetch to a file and sanity-check it rather
    # than piping straight into bash, so an HTML error page can't execute.
    local tmp; tmp=$(mktemp)
    curl -fsSL https://claude.ai/install.sh -o "$tmp" || { rm -f "$tmp"; return 1; }
    head -c 20 "$tmp" | grep -q '<' && { bad "installer returned HTML, not a script"; rm -f "$tmp"; return 1; }
    bash -n "$tmp" 2>/dev/null || { bad "installer failed syntax check"; rm -f "$tmp"; return 1; }
    bash "$tmp" >/dev/null 2>&1; local rc=$?
    rm -f "$tmp"; return $rc
}

if command -v claude >/dev/null 2>&1; then
    ok "claude present ($(command -v claude), $(claude --version 2>/dev/null | head -1))"
else
    warn "installing Claude Code via signed apt repo ..."
    if install_claude_apt && command -v claude >/dev/null 2>&1; then
        ok "claude installed from apt (verified signing key)"
    else
        warn "apt path unavailable; trying the native installer ..."
        if install_claude_native && command -v claude >/dev/null 2>&1; then
            ok "claude installed to ~/.local/bin/claude"
        else
            bad "both install paths FAILED. Try manually and read the error:"
            bad "  curl -fsSL https://claude.ai/install.sh | bash"
            bad "  https://code.claude.com/docs/en/troubleshoot-install"
        fi
    fi
fi

# ~/.profile only runs for login shells; make sure non-login shells (tmux, cron)
# find the binary too.
if echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin" \
   || grep -q 'local/bin' "$HOME/.bashrc" 2>/dev/null; then
    ok "~/.local/bin on PATH"
else
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    ok "added ~/.local/bin to PATH in ~/.bashrc"
fi

# `claude`-launched local background agents do not survive an SSH connection
# dropping — they run under a supervisor tied to the SSH session's process
# tree. tmux persists server-side independent of SSH, so wrap `claude` in a
# shell function that transparently runs it inside a named tmux session:
# typing `claude` after a drop reattaches you to right where you left off.
# A function (not an alias) so it can inspect $TMUX/args before deciding.
if grep -q 'claude() { # tmux-persistent wrapper' "$HOME/.bashrc" 2>/dev/null; then
    ok "claude tmux wrapper already in ~/.bashrc"
else
    cat >> "$HOME/.bashrc" <<'BASHRC_EOF'
claude() { # tmux-persistent wrapper (reattach after a dropped SSH connection)
    # Already nested in tmux, not an interactive terminal (piped/scripted use
    # like `claude -p "..."`), or tmux missing entirely: just run it directly.
    if [ -n "${TMUX:-}" ] || [ ! -t 1 ] || ! command -v tmux >/dev/null 2>&1; then
        command claude "$@"
        return
    fi
    # One session per project directory, so different repos don't collide.
    local session="claude-$(basename "$PWD")"
    if tmux has-session -t "$session" 2>/dev/null; then
        # Reattaching ignores any new args passed on THIS call — the session
        # is already running with whatever args started it. To force a fresh
        # one: tmux kill-session -t "$session"
        tmux attach-session -t "$session"
    else
        local cmd="command claude" arg
        for arg in "$@"; do cmd="$cmd $(printf '%q' "$arg")"; done
        tmux new-session -d -s "$session" -c "$PWD" "$cmd"
        tmux attach-session -t "$session"
    fi
}
BASHRC_EOF
    ok "added tmux-persistent claude wrapper to ~/.bashrc"
fi

# --------------------------------------------------------------------------
step "3. Claude sessions + memory (restored by copy)"
restore_claude
warn "on a fresh instance you must still log in to Claude (and gh, if you use it)"

# --------------------------------------------------------------------------
step "4. Repository"
# The repo is PUBLIC, so an anonymous HTTPS clone works with no credentials on
# the box — which is the point: a fresh big-GPU instance has no gh login, no
# PAT, and no volume. Push still needs auth (step 5), but pull does not.
if [ -d "$REPO/.git" ]; then
    ok "repo present at $REPO (branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
    # Only fast-forward. Never touch a dirty tree or discard local commits:
    # on the home box this script runs while real work is checked out.
    if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
        warn "working tree is dirty — skipping fetch/update (nothing is ever discarded)"
    else
        git -C "$REPO" fetch --quiet origin 2>/dev/null \
            && ok "fetched origin" || warn "fetch failed (offline?) — using the tree as-is"
    fi
else
    warn "no repo at $REPO — cloning $BRANCH ..."
    # --branch pins the checkout at clone time so we never land on main, which
    # lacks the snapshot loader and configs/exp/ entirely.
    if git clone --quiet --branch "$BRANCH" "$REPO_URL" "$REPO" 2>/dev/null; then
        ok "cloned $BRANCH -> $REPO"
    elif git clone --quiet "$REPO_URL" "$REPO" 2>/dev/null; then
        # Branch may not exist on the remote; land the clone, then say so loudly
        # rather than silently running the wrong code.
        bad "branch '$BRANCH' not found on origin — cloned the default branch instead"
        bad "the exp configs and the snapshot dataset loader are probably MISSING"
    else
        bad "clone FAILED from $REPO_URL"
        bad "check network, or set REPO_URL if the repo moved"
        exit 1
    fi
fi

# Sanity-check that the checkout actually carries what an experiment needs.
# A clone that succeeds but lands on the wrong branch is the failure mode worth
# catching here, not three minutes later inside a run.
N_CFG=$(ls "$REPO/configs/exp"/*.py 2>/dev/null | wc -l)
[ "$N_CFG" -gt 0 ] && ok "configs/exp present ($N_CFG configs)" \
                   || bad "no configs in configs/exp — wrong branch? (want $BRANCH)"
[ -f "$REPO/datasets/Snapshots/log_103_comparison_200.json" ] \
    && ok "log_103 snapshot dataset present" \
    || warn "log_103 snapshot dataset missing — snapshot-based configs will fail"
grep -q 'register(DATASETS, "snapshot")' "$REPO/src/bias_steer/datasets.py" 2>/dev/null \
    && ok "snapshot dataset loader registered" \
    || warn "no snapshot loader in datasets.py — wrong branch?"

# --------------------------------------------------------------------------
step "5. Git push access"
# Test the CAPABILITY, not the tool. What actually matters is whether git can
# authenticate to origin; gh is one way to arrange that, not a requirement.
# (gh is a Go binary with no pip equivalent — the `gh` package on PyPI is an
# unrelated project at v0.0.4 — so it can never live in the venv.)
if git -C "$REPO" ls-remote origin >/dev/null 2>&1; then
    ok "git can authenticate to origin"
    # gh stores its helper HOST-SCOPED as credential.https://github.com.helper,
    # not as a bare credential.helper — querying the wrong key made an earlier
    # version of this script re-run `gh auth setup-git` on every boot.
    # git can print the same key twice, once with an empty value; take the first
    # line that actually HAS a value and strip the key off it.
    helper=$(git config --get-regexp 'credential\..*\.helper' 2>/dev/null \
             | awk 'NF>1 {$1=""; sub(/^ /,""); print; exit}')
    [ -n "$helper" ] && ok "credential helper: $helper"
else
    bad "git CANNOT authenticate to origin. Either:"
    bad "  A) gh:  gh auth login && gh auth setup-git"
    bad "  B) PAT: git config --global credential.helper store && git push"
    bad "     (paste a fine-grained PAT with Contents:read+write as the password;"
    bad "      stored in plaintext at ~/.git-credentials, so prefer A)"
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    ok "gh authenticated as $(gh api user --jq .login 2>/dev/null || echo '?')"
elif command -v gh >/dev/null 2>&1; then
    warn "gh installed but not authenticated (optional — only needed for PRs/issues)"
fi

# A commit with no configured identity fails outright ("Author identity
# unknown, please tell me who you are") — this bit an earlier session on a
# fresh box mid-task. Set it globally, once, so every box this script
# provisions can commit without that surprise.
GIT_NAME="DarksharkThe1st"
GIT_EMAIL="kitturfarhan@gmail.com"
if [ "$(git config --global --get user.name 2>/dev/null)" = "$GIT_NAME" ] \
   && [ "$(git config --global --get user.email 2>/dev/null)" = "$GIT_EMAIL" ]; then
    ok "git identity: $GIT_NAME <$GIT_EMAIL>"
else
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    ok "set git identity: $GIT_NAME <$GIT_EMAIL>"
fi

# --------------------------------------------------------------------------
step "6. Venv, secrets, GPU"
# Runtime pins, taken from the venv that actually produced the Aug 9 results
# (`.venv/bin/pip freeze`). Embedded here rather than in a requirements file
# because this script travels alone — there are no push credentials on the home
# box, so a new file could not be committed for the clone to pick up.
#
# Deliberately the RUNTIME set only, not the full 179-package freeze: setup.py
# also pulls jupyter/plotly/ipython, which a headless run never imports and
# which are the most likely things to fail resolution on a different box. pip
# resolves the transitive deps of these itself.
PINS=(
    "torch==2.13.0"
    "transformers==5.14.1"
    "transformer-lens==3.7.0"
    "transformers-stream-generator==0.0.5"
    "accelerate==1.14.0"
    "huggingface_hub==1.27.0"
    "datasets==5.0.1"
    "safetensors==0.8.0"
    "sentencepiece==0.2.2"
    "einops==0.8.2"
    "jaxtyping==0.3.7"
    "numpy==2.2.6"
    "pandas==2.3.3"
    "openai==2.53.0"
    "tqdm==4.70.0"
)

build_venv() {
    # setup.py declares python_requires=">=3.12", but the egg-info from the
    # install that produced every existing result records ">=3.10" and that venv
    # is 3.10.12 — the constraint was tightened after the fact and the code
    # demonstrably runs on 3.10. Prefer a new enough interpreter when the box has
    # one; otherwise fall back and tell pip to ignore the stale bound rather than
    # failing the whole bootstrap on a cosmetic mismatch.
    local py=""
    for cand in python3.12 python3.11 python3; do
        command -v "$cand" >/dev/null 2>&1 && { py="$cand"; break; }
    done
    [ -n "$py" ] || { bad "no python3 on this box"; return 1; }
    ok "using $py ($($py --version 2>&1))"

    "$py" -m venv "$REPO/.venv" 2>/dev/null || { bad "venv creation FAILED (need python3-venv)"; return 1; }
    local pip="$REPO/.venv/bin/pip"
    "$pip" install --quiet --upgrade pip >/dev/null 2>&1

    warn "installing pinned deps (~6GB, several minutes — torch is most of it) ..."
    "$pip" install --quiet "${PINS[@]}" || { bad "dep install FAILED"; return 1; }

    # The package itself, --no-deps because PINS already covers what it needs and
    # we do not want setup.py re-resolving jupyter et al.
    "$pip" install --quiet --no-deps -e "$REPO" 2>/dev/null \
        || "$pip" install --quiet --no-deps --ignore-requires-python -e "$REPO" \
        || { bad "editable install of the package FAILED"; return 1; }
    return 0
}

if [ ! -x "$REPO/.venv/bin/python" ] && [ "$PORTABLE" = 1 ]; then
    warn "no venv — building one (portable mode)"
    build_venv || bad "venv build failed; fix the error above and re-run"
fi

if [ -x "$REPO/.venv/bin/python" ]; then
    # The venv hardcodes absolute shebangs, so it only works if the volume mounts
    # at the same path. Prove it imports rather than merely exists.
    if "$REPO/.venv/bin/python" -c 'import torch, transformer_lens' >/dev/null 2>&1; then
        ok "venv works ($("$REPO/.venv/bin/python" -c 'import torch; print("torch "+torch.__version__)' 2>/dev/null))"
        "$REPO/.venv/bin/python" -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
            && ok "torch sees CUDA" \
            || bad "torch cannot see CUDA — driver mismatch; recreate the venv"
    else
        bad "venv exists but imports FAIL — changed mount path or driver."
        bad "  Recreate: cd $REPO && python3 -m venv .venv && .venv/bin/pip install -e ."
    fi
else
    bad "no venv — create: cd $REPO && python3 -m venv .venv && .venv/bin/pip install -e ."
    bad "  (or re-run this script; portable mode builds it automatically)"
fi

# .env is gitignored (.gitignore:141), so it is NEVER in the clone and the key
# must be entered by hand on this box.
#
# By design this script will NOT accept the key by any automated route — not an
# env var, not a copied file, not an argument. Reasons, in order of how likely
# they are to bite:
#   1. An env var on the ssh command line lands in the mac's shell history and is
#      readable via `ps` by any user on the box while the script runs.
#   2. Copying .env between machines multiplies the number of disks holding a
#      live credential, and each copy outlives the instance you made it for.
#   3. Any .bak/.orig/.tmp sibling of .env is NOT covered by .gitignore, which
#      lists only `.env` and `.envrc` — so a copy inside the repo is committable
#      even though the original is not.
# Pasting a fresh key into a fresh .env is one extra step and has none of these
# properties. Rotate the old key at platform.openai.com/api-keys instead.
if [ ! -f "$REPO/.env" ]; then
    warn "no .env (gitignored, so the clone never has one) — creating"
    umask 077   # this file will hold a live credential; 600 from birth
    printf 'HF_HOME=%s\nOPENAI_API_KEY=\n' "$HF_CACHE" > "$REPO/.env"
    umask 022
    ok "wrote .env with HF_HOME (mode 600)"
    bad "OPENAI_API_KEY is EMPTY — paste a fresh key in before running:"
    bad "  nano $REPO/.env      # or vim; set OPENAI_API_KEY=sk-..."
    bad "Then rotate/revoke the old key: https://platform.openai.com/api-keys"
fi

if [ -f "$REPO/.env" ]; then
    # Match a NON-EMPTY value: the placeholder line we just wrote would satisfy a
    # bare `grep OPENAI_API_KEY=` and report a key that is not there.
    if grep -qE '^OPENAI_API_KEY=.+' "$REPO/.env"; then
        ok ".env has OPENAI_API_KEY"
    else
        bad ".env has no OPENAI_API_KEY value — the judge will fail on the first batch"
        bad "  paste a fresh key:  nano $REPO/.env"
    fi
    hf=$(grep -oP '(?<=^HF_HOME=).*' "$REPO/.env" 2>/dev/null)

    # A .env copied off the home box points HF_HOME at the volume, which does not
    # exist here and cannot be created (no write access to /lambda). Left alone,
    # HF fails partway into the first model load instead of at bootstrap. Repoint
    # it at local disk; keep a .bak so the original is recoverable.
    # Kept as a safety net for a hand-copied .env, but edited IN PLACE with no
    # backup file: a .env.bak here would hold the key and is NOT gitignored
    # (.gitignore lists only .env and .envrc), so it would be committable.
    if [ "$PORTABLE" = 1 ] && [ -n "$hf" ] && [ ! -d "$hf" ] && [ "$hf" != "$HF_CACHE" ]; then
        sed -i "s|^HF_HOME=.*|HF_HOME=$HF_CACHE|" "$REPO/.env"
        hf="$HF_CACHE"
        warn "HF_HOME pointed at a path that does not exist here (the volume)"
        ok "repointed HF_HOME -> $HF_CACHE (edited in place; no .bak written)"
    fi

    if [ -n "$hf" ]; then
        if [ -d "$hf/hub" ]; then
            ok "HF cache at $hf ($(du -sh "$hf" 2>/dev/null | cut -f1))"
        elif [ "$PORTABLE" = 1 ]; then
            # Expected on a fresh box: weights are not in git and never will be.
            warn "HF_HOME=$hf is empty — first run downloads the weights"
            warn "  qwen-7b is ~15GB; it lands on the EPHEMERAL root disk"
        else
            bad "HF_HOME=$hf but no hub/ there — models would re-download (~56GB)"
        fi
    else
        warn "no HF_HOME in .env — models would cache to the EPHEMERAL root disk"
    fi
    # `echo x >> .env` silently concatenates onto a file with no trailing
    # newline, which once corrupted the API key. Fix it pre-emptively.
    [ -n "$(tail -c1 "$REPO/.env")" ] && { printf '\n' >> "$REPO/.env"; warn ".env had no trailing newline — added one"; }
else
    bad "no .env at $REPO/.env (cp .env.example .env and fill it in)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    if [ "$NGPU" -gt 1 ]; then
        # Worth stating plainly: models.py loads via
        # HookedTransformer.from_pretrained_no_processing(..., device=device)
        # where device is the bare string "cuda" from get_device(). That is GPU 0
        # only — there is no device_map, no sharding, no DataParallel anywhere in
        # the codebase. Extra GPUs sit idle. To use them, run N separate configs
        # concurrently with CUDA_VISIBLE_DEVICES pinned per process.
        warn "$NGPU GPUs visible, but transformer_lens loads onto ONE (device=\"cuda\" = GPU 0)"
        warn "  extra GPUs are idle; for parallelism run one config per GPU:"
        warn "  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.bias_steer run configs/exp/a.py &"
        warn "  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.bias_steer run configs/exp/b.py &"
    fi

    # Batch-size advice for a 7B. Anchored on a MEASURED point, not theory: the
    # Aug 9 qwen-7b runs completed at batch_size=32 on a 40GB A100. fp16 weights
    # are ~15GB, so ~25GB served 32 items => ~0.8GB/item for KV cache +
    # activations at max_tokens=128. Scale that ratio and round DOWN to a power
    # of two for headroom.
    if [ "${VRAM_MB:-0}" -gt 0 ]; then
        FREE_GB=$(( VRAM_MB / 1024 - 15 ))
        if [ "$FREE_GB" -le 4 ]; then
            REC=8
        else
            EST=$(( FREE_GB * 32 / 25 ))
            REC=8; for cand in 16 32 64 128 256; do [ "$EST" -ge "$cand" ] && REC=$cand; done
        fi
        printf '  \033[32m✓\033[0m 7B sizing: ~%s GB free after fp16 weights -> batch_size \033[1m%s\033[0m\n' \
            "$FREE_GB" "$REC"
        printf '      (measured anchor: batch_size=32 fit a 40GB A100 at max_tokens=128)\n'
        if [ "$REC" -gt 32 ]; then
            printf '      raise it in the config, e.g.:\n'
            printf '        sed -i "s/batch_size=32/batch_size=%s/" %s/configs/exp/anchor_qwen7b.py\n' \
                "$REC" "$REPO"
        fi
    fi
else
    bad "nvidia-smi missing — no GPU visible"
fi

# --------------------------------------------------------------------------
step "7. Storage"
LOG="$OPS/storage-log.tsv"
[ -f "$LOG" ] || printf 'timestamp_utc\tvolume_gb\troot_used_gb\troot_avail_gb\thf_cache_gb\tvenv_gb\truns_gb\tclaude_gb\n' > "$LOG"

gb() { du -sBG "$1" 2>/dev/null | cut -f1 | tr -d 'G' || echo 0; }
VOL_GB=$(gb "$BASE")
HF_GB=$(gb "$HF_CACHE")
VENV_GB=$(gb "$REPO/.venv")
RUNS_GB=$(gb "$REPO/runs")
CLAUDE_GB=$(gb "$CLAUDE_STORE")
ROOT_USED=$(df -BG / | awk 'NR==2{gsub("G","",$3); print $3}')
ROOT_AVAIL=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$VOL_GB" "$ROOT_USED" "$ROOT_AVAIL" \
    "$HF_GB" "$VENV_GB" "$RUNS_GB" "$CLAUDE_GB" >> "$LOG"

if [ "$PORTABLE" = 1 ]; then
    printf '  install root (EPHEMERAL): %s   %s GB used   (threshold %s GB)\n' \
        "$BASE" "$VOL_GB" "$THRESHOLD_GB"
else
    printf '  persistent volume : %s GB   (threshold %s GB)\n' "$VOL_GB" "$THRESHOLD_GB"
fi
printf '    ├─ hf_cache     : %s GB\n' "$HF_GB"
printf '    ├─ .venv        : %s GB\n' "$VENV_GB"
printf '    ├─ runs/        : %s GB\n' "$RUNS_GB"
printf '    └─ .claude-home : %s GB\n' "$CLAUDE_GB"
printf '  root disk (EPHEMERAL): %s GB used, %s GB free\n' "$ROOT_USED" "$ROOT_AVAIL"
printf '  history: %s\n' "$LOG"

ALERT="$OPS/STORAGE_ALERT.md"
if [ "${VOL_GB:-0}" -gt "$THRESHOLD_GB" ]; then
    cat > "$ALERT" <<ALERTEOF
# STORAGE ALERT — cleanup needed

The persistent volume is at **${VOL_GB} GB**, over the ${THRESHOLD_GB} GB threshold.
Generated $(date -u '+%Y-%m-%d %H:%M UTC') by bootstrap.sh. Delete this file once resolved.

## Instructions for Claude

Storage is over threshold. Work through this list in order, **reporting the space
reclaimed at each step and stopping once under ${THRESHOLD_GB} GB**. Confirm with the
user before any deletion not listed as safe.

### Safe to delete without asking
1. **Residual tensors** — \`$REPO/runs/*/residuals.safetensors\`. Bulky, gitignored by
   design, regenerable from committed inputs. Usually the biggest single win.
   \`find $REPO/runs -name residuals.safetensors -delete\`
2. **Discarded runs** — anything under \`$REPO/runs/_discard/\`. Explicitly throwaway.
3. **Package caches** — \`~/.cache/pip\`, \`~/.npm\`. Note these live on the EPHEMERAL root
   disk, so they only help if the root disk is what is full, not the volume.

### Ask first — recoverable but costly to redo
4. **Unused model weights** in \`$HF_CACHE/hub/\`. Check which models the current
   configs reference before removing any; Qwen1.5-14B alone is ~28GB.
   Inspect: \`du -sh $HF_CACHE/hub/*\`
   Cross-check: \`grep -rh 'models=' $REPO/configs/\`
5. **Old session transcripts** in \`$CLAUDE_STORE/projects/*/\`. This is the user's Claude
   history, which they explicitly asked to persist — treat as precious, prune only with
   explicit approval.

### Never delete
- \`$REPO/.git\` — carries every committed run and result.
- \`$REPO/.env\` — API keys, not recoverable from anywhere.
- \`$REPO/runs/*/\` beyond residuals — results, logs, vectors and manifests are the
  scientific record.
- \`$REPO/.venv\` unless rebuilding it immediately (~6GB, ~10min to restore).

### After cleanup
Re-run \`bash $VOL/bootstrap.sh\` to confirm, then delete this alert file.
ALERTEOF
    printf '\n'
    bad "OVER THRESHOLD: ${VOL_GB} GB > ${THRESHOLD_GB} GB"
    bad "Cleanup instructions written to $ALERT"
    bad "Tell Claude: \"read $ALERT and do the cleanup\""
else
    [ -f "$ALERT" ] && { rm -f "$ALERT"; ok "back under threshold — cleared stale alert"; }
    ok "under threshold"
fi

# --------------------------------------------------------------------------
step "8. Back up Claude state"
backup_claude

step "Done"
echo "  cd $REPO"
echo
if [ "$PORTABLE" = 1 ]; then
    echo "  Run the identical-config pair (same seed, back to back):"
    echo "    cd $REPO"
    echo "    .venv/bin/python -m src.bias_steer run configs/exp/anchor_qwen7b.py"
    echo "    .venv/bin/python -m src.bias_steer run configs/exp/anchor_qwen7b.py"
    echo
    echo "  Then diff the two run folders (run_id is a timestamp, so they differ):"
    echo "    ls -dt runs/*anchor-qwen-7b* | head -2"
    echo "    diff <(cut -d, -f2- runs/<A>/results.csv) <(cut -d, -f2- runs/<B>/results.csv)"
    echo
    echo "  ROOT DISK IS EPHEMERAL — copy results off before terminating:"
    echo "    scp -r ubuntu@<this-ip>:$REPO/runs ."
else
    echo "  Sessions only reach the volume when a backup runs. To automate hourly:"
    echo "    (crontab -l 2>/dev/null; echo '0 * * * * bash $VOL/bootstrap.sh --backup >/dev/null 2>&1') | crontab -"
fi
echo
