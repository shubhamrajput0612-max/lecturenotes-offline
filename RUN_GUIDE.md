# Run Guide: LectureNotes Offline

Follow these steps on your own laptop (Windows, Python 3.10 or newer).

## 1. Install dependencies
```
pip install -r requirements.txt
```
- `llama-cpp-python` may need to compile. If it fails on Windows, install
  "Build Tools for Visual Studio" (C++ workload) and try again.
- The first Whisper run downloads the speech model, so it needs internet once.

## 2. Get a language model file
Download a **Llama 3.2 3B Instruct** model in `.gguf` format (a `Q4_K_M` file is a
good size) and put it in a `models/` folder.

## 3. Run on a short test clip
```
python app.py --audio lecture.mp3 --lang gu --llm-model models/<your-model-file>.gguf
```
Use `--lang hi` for Hindi or `--lang en` for English. Try a 1-2 minute clip first.

## 4. Check the output
The `output/` folder should contain `transcript.txt`, `notes.md` and `flashcards.csv`.

## 5. Update the README
Tick only what you actually tested in Section 6 of `README.md`.

## Common problems
- **"LLM file not found":** check the path after `--llm-model`.
- **Empty transcript:** the audio may be silent or too noisy. Try another clip.
- **Very slow:** use `--whisper-size base` or `tiny`, and a shorter clip.
