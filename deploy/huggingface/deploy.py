#!/usr/bin/env python3
import argparse
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def run(command, cwd=REPO_ROOT, *, capture=False, env=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def require_clean_worktree():
    status = run(["git", "status", "--porcelain"], capture=True)
    if status:
        raise SystemExit(
            "Working tree is not clean. Commit or stash changes before deploying."
        )


def remote_url(remote):
    return run(["git", "remote", "get-url", remote], capture=True)


def deployment_remote_url(remote):
    return os.getenv("HF_SPACE_REMOTE_URL") or remote_url(remote)


def git_auth_env(target_dir):
    username = os.getenv("HF_DEPLOY_USERNAME")
    token = os.getenv("HF_DEPLOY_TOKEN")
    if not os.getenv("HF_SPACE_REMOTE_URL"):
        return None
    if not username or not token:
        raise SystemExit(
            "HF_SPACE_REMOTE_URL requires HF_DEPLOY_USERNAME and HF_DEPLOY_TOKEN."
        )

    askpass_path = target_dir / ".git" / "hf-askpass.sh"
    askpass_path.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "*Username*) printf '%s\\n' \"$HF_DEPLOY_USERNAME\" ;;\n"
        "*) printf '%s\\n' \"$HF_DEPLOY_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass_path.chmod(0o700)

    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def export_current_tree(target_dir):
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target_dir)


def apply_huggingface_overlay(target_dir):
    deploy_dir = target_dir / "deploy"
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)

    shutil.copyfile(SCRIPT_DIR / "Dockerfile", target_dir / "Dockerfile")

    header = (SCRIPT_DIR / "README.header.yml").read_text(encoding="utf-8").strip()
    readme_path = target_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(f"{header}\n\n{readme}", encoding="utf-8")


def commit_and_push(target_dir, remote, branch, message, dry_run):
    url = deployment_remote_url(remote)

    run(["git", "init"], cwd=target_dir)
    git_env = git_auth_env(target_dir)
    run(["git", "config", "user.name", "Notes Chat Deploy"], cwd=target_dir)
    run(["git", "config", "user.email", "deploy@example.invalid"], cwd=target_dir)
    run(["git", "remote", "add", remote, url], cwd=target_dir)
    run(["git", "fetch", "--depth=1", remote, branch], cwd=target_dir, env=git_env)
    run(["git", "reset", "--mixed", "FETCH_HEAD"], cwd=target_dir)
    run(["git", "add", "-A"], cwd=target_dir)

    staged = run(["git", "diff", "--cached", "--stat"], cwd=target_dir, capture=True)
    if not staged:
        print("No deployment changes to push.")
        return

    print(staged)

    if dry_run:
        print(f"Dry run complete. Would push to remote '{remote}' branch '{branch}'.")
        return

    run(["git", "commit", "-m", message], cwd=target_dir)
    run(
        [
            "git",
            "-c",
            "http.postBuffer=524288000",
            "push",
            remote,
            f"HEAD:{branch}",
        ],
        cwd=target_dir,
        env=git_env,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and push the Hugging Face Space deployment overlay."
    )
    parser.add_argument("--remote", default="hf")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Deploy notes chat app")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    require_clean_worktree()

    with tempfile.TemporaryDirectory(prefix="notes-chat-hf-") as temp:
        target_dir = Path(temp)
        export_current_tree(target_dir)
        apply_huggingface_overlay(target_dir)
        commit_and_push(
            target_dir,
            remote=args.remote,
            branch=args.branch,
            message=args.message,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
