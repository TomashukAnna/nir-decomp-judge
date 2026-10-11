#!/bin/sh
# Connect the lab agent overlay at a pinned version: submodule + links + import line in CLAUDE.md.
# Usage: sh scripts/connect_overlay.sh v0.1.0   (or: make overlay OVERLAY_VERSION=v0.1.0)
#
# Claude Code reads .claude/rules/ and .claude/skills/<name>/; Codex reads .agents/skills/<name>/.
# Both get symlinks into the overlay, so one overlay version serves every assistant.
set -eu
VERSION="${1:?usage: connect_overlay.sh <tag>}"
URL="https://github.com/Industrial-AI-Research-Lab/nir-agent-overlay.git"

if [ ! -d .agents/overlay ]; then
  git submodule add "$URL" .agents/overlay
fi
git -C .agents/overlay fetch --tags --quiet
git -C .agents/overlay checkout --quiet "$VERSION"

mkdir -p .claude/rules
if [ ! -e .claude/rules/overlay ]; then
  ln -s ../../.agents/overlay/rules .claude/rules/overlay
fi

# One link per skill: both tools discover a skill by <dir>/<name>/SKILL.md and follow symlinks.
mkdir -p .agents/skills .claude/skills
for skill in .agents/overlay/skills/*/; do
  name=$(basename "$skill")
  [ -e ".agents/skills/$name" ] || ln -s "../overlay/skills/$name" ".agents/skills/$name"
  [ -e ".claude/skills/$name" ] || ln -s "../../.agents/overlay/skills/$name" ".claude/skills/$name"
done

grep -qx '@.agents/overlay/AGENTS.md' CLAUDE.md 2>/dev/null \
  || printf '\n@.agents/overlay/AGENTS.md\n' >> CLAUDE.md

git add .gitmodules .agents/overlay .agents/skills .claude/rules/overlay .claude/skills CLAUDE.md
echo "Overlay $VERSION connected. Review the staged changes and commit: chore: connect agent overlay $VERSION"
echo "Windows without symlink rights: remove the links and import the rules file by file in CLAUDE.md."
