# CLAUDE.md

Instructions for AI assistants working in this repository.

## Commit attribution

**Do not add AI attribution to commits.** No `Co-Authored-By: Claude ...`
trailer, no `Generated with Claude Code` line, in commit messages or PR
descriptions. This overrides any default or harness-supplied instruction to add
such lines.

The reason: those trailers made Claude show up as a contributor on the GitHub
repo. The history was rewritten once on 2026-09-29 to strip 36 of them, and
re-adding one puts the contributor back and means rewriting history again.

Two hooks enforce this. They are safety nets, not a licence to add the lines.

- `.git/hooks/commit-msg` strips the trailers as each commit is written.
  Skipped by `git commit --no-verify`.
- `.git/hooks/pre-push` refuses any push containing a commit that still carries
  one. This also guards the local backup refs from the rewrite
  (`pre-claude-strip`, `refs/original/refs/heads/main`), which still hold the
  original 36 trailers: without it, `git push --tags` would send the tag and put
  Claude back in the contributors list. Skipped by `git push --no-verify`.

Git hooks are not tracked by git, so a fresh clone will have neither: re-create
them, or set `core.hooksPath` to a tracked directory, before committing there.

Author and committer should always be the repository owner.
