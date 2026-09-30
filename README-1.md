# LectureNotes Offline: Gujarati/Hindi Lecture Notes on Snapdragon

**An offline, on-device tool that turns recorded lectures (Gujarati, Hindi or English) into a transcript, a summary and flashcards. No internet and no cloud upload needed.**

Built for the **Snapdragon® AI Lab Build & Present Challenge 2026** (Qualcomm).
Author: Rajput Shubham, B.E. Sem-I, Government Engineering College, Modasa (GTU)
Target hardware: Snapdragon-powered HP AI PC (Snapdragon X series, Hexagon NPU), Windows 11 ARM64

---

## 1. The problem

Many engineering students in India attend lectures that mix Gujarati, Hindi and English technical terms. Taking notes while listening is hard, and most AI note-taking tools:

- need a stable internet connection, which campus Wi-Fi and mobile data often don't provide;
- upload recordings to servers abroad, which raises privacy and cost concerns;
- work best in English and handle Gujarati or Hindi poorly.

**Goal:** a tool a student can run on their own laptop, offline, that takes a lecture recording and gives back study material they can revise from.

## 2. What it does

```
Lecture audio (.wav / .mp3 / .m4a)
        │
        ▼
 [1] Speech-to-text  (Whisper)          → transcript (Gujarati / Hindi / English)
        │
        ▼
 [2] Summary + key points (small LLM)   → short revision notes
        │
        ▼
 [3] Flashcards (small LLM)             → Q&A cards
        │
        ▼
 [4] Export                             → notes.md / flashcards.csv
```

## 3. Models used

| Stage | Model | Source | Precision |
|---|---|---|---|
| Speech-to-text | **Whisper** (`Whisper-Base` or `Whisper-Small`, multilingual) | Qualcomm AI Hub (`qai-hub-models`) | quantized (w8a16) |
| Summary and flashcards | **Llama 3.2 3B Instruct** | Qualcomm AI Hub (optimized for Snapdragon) | quantized |

> `<EDIT: write the exact model names and variants you actually used in the final build.>`

Why these: Whisper supports Hindi and Gujarati, and both models are published for Snapdragon on Qualcomm AI Hub, so the deployment path is a compile step and not a rewrite.

## 4. How it runs on the NPU

1. **Model source:** models come from Qualcomm AI Hub (`pip install qai-hub qai-hub-models`), using your AI Hub API token.
2. **Compile and profile:** AI Hub compiles the model for the target Snapdragon device and reports which layers run on the Hexagon NPU, plus latency and memory.
3. **On-device inference:** the app runs the compiled model through **ONNX Runtime with the QNN Execution Provider** (or the Qualcomm runtime for the AI Hub export), which sends the work to the Hexagon NPU.
4. **CPU fallback:** if the NPU path is unavailable, the app falls back to a CPU model and prints which engine is active, so it never claims NPU use when it isn't running there.

**Why the NPU:** transcribing an hour of audio and running an LLM are heavy jobs. The NPU is designed to do them at low power, which matters for battery life during a full day of lectures.

## 5. Setup and usage

```bash
# 1. Clone
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# 2. Install (Windows 11 ARM64 or any machine for the CPU path)
pip install -r requirements.txt

# 3. (Optional) Compile/profile for the NPU with your AI Hub token
export QAI_HUB_API_TOKEN="<your token from app.aihub.qualcomm.com>"
python qualcomm_npu.py --device "Snapdragon X Elite CRD"

# 4. Run on a lecture recording
python app.py --audio lecture.mp3 --lang gu   # gu = Gujarati, hi = Hindi, en = English
```

Output: `output/transcript.txt`, `output/notes.md`, `output/flashcards.csv`

> `<EDIT: match these commands to your actual files and arguments.>`

## 6. What I tested and what I didn't

Please read this section. I have tried to be honest about what is verified.

### Tested
- [ ] `<Pipeline runs end to end on a sample recording on my machine (state which machine, e.g. "Windows laptop, CPU, x86")>`
- [ ] `<Transcription tried on: Gujarati / Hindi / English sample(s), length ___ minutes>`
- [ ] `<Summary and flashcards generated and read through by me for correctness>`

### Not tested / not verified
- [ ] `<Execution on a real Snapdragon HP AI PC. If you had no access, say so here.>`
- [ ] `<NPU latency, power or memory numbers. No NPU performance is claimed unless you measured it.>`
- [ ] `<Long recordings (1 hour or more)>`
- [ ] `<Noisy classroom audio, multiple speakers>`

### Known limitations
- Gujarati transcription is usually less accurate than Hindi or English, especially with technical terms and code-mixed speech (Gujarati + English words).
- A 3B-parameter LLM can write weaker or less accurate summaries in Gujarati than in English. Summaries and flashcards should be reviewed before studying from them.
- Whisper can make mistakes on names, formulas and equations.
- The CPU fallback is slower than the NPU path.

## 7. Privacy

All processing is designed to happen on the device. Audio, transcripts and notes are never sent to any server, and the app needs no network connection once the models are downloaded.

## 8. Future work

- Add speaker separation for lectures with questions from students
- Pull formulas and definitions out into a separate "formula sheet"
- Export flashcards directly to Anki
- Subject-specific glossary for GTU subjects (e.g. Basic Electronics)

## 9. Acknowledgements

Qualcomm® AI Hub and Snapdragon® platform, OpenAI Whisper, Meta Llama 3.2, ONNX Runtime. Snapdragon and Qualcomm are trademarks of Qualcomm Incorporated.

## 10. License

`<Choose one, e.g. MIT>`
