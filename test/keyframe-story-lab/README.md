# Keyframe Story Lab (deepseek-chat)

Small test harness to evaluate keyframe-driven story progression in a chat-only format.
It uses two prompt phases:
- Judge: classifies player intent and decides if a keyframe advances.
- Narrator: replies in-character based on the current frame and judge output.

## Quick start

1) Make sure `DEEPSEEK_APIKEY` is set in `.env` at the repo root.
2) Run a scripted test:

```
python test/keyframe-story-lab/app.py --script test/keyframe-story-lab/sample_inputs.txt
```

3) Or run interactively:

```
python test/keyframe-story-lab/app.py --interactive
```

## Files

- `test/keyframe-story-lab/app.py`: runner
- `test/keyframe-story-lab/keyframes.json`: keyframe definitions
- `test/keyframe-story-lab/sample_inputs.txt`: sample player inputs

## Notes

- The runner reads `./.cache/dialogs.md` as style reference if present.
- Use `--dry-run` to print the assembled prompts without calling the API.
- Override the API base URL via `--base-url` or `DEEPSEEK_BASE_URL` if needed.
- Bracketed player messages default to emote-only (`--bracket-mode emote`), and world edits are rejected locally.
- Use `--bracket-mode strict` to reject all bracketed text, or `--bracket-mode allow` to disable this check.
- Aliya sends an opening line before any COSMOS input. Use `--opening-mode auto|model|static|none` to control it.
- Each keyframe can define `guidance` hints to steer the player when they go off-track.
