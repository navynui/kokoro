# TODO — Add MMS-TTS + ThonburianTTS (F5) Thai engines

- [x] **Phase 0: Research** — verified both options (MMS works in container; Thonburian needs deps/ffmpeg/ref voice)
- [x] **Phase 1: MMS engine** — `mms` in server.py, voices, UI, `/voices`
- [x] **Phase 2: Thonburian engine** — Dockerfile (ffmpeg + deps), `f5` in server.py, default ref voice
- [x] **Phase 3: Build & test** — rebuild container, test mms + f5 + kokoro + thai + piper regression
- [x] **Phase 4: Docs & commit** — README/AGENTS/SKILL updates, git add/commit/push

## Progress

- Phase 1: MMS (verified live: 0.33s synth, no new deps)
- Phase 2: Thonburian (heavy: 1.35GB model, ffmpeg, f5-tts/vocos/librosa deps)
