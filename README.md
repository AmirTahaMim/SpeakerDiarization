# Speaker Diarization and Audio Splitting

This project provides an automated pipeline for **speaker diarization** and **audio splitting** using a pre-trained model from Hugging Face's `pyannote.audio` library. The system processes an input audio file, identifies speakers, and saves their speech segments as separate `.wav` files.

## Features
- **Speaker Diarization**: Automatically detects and labels different speakers in an audio file.
- **Audio Splitting**: Extracts and saves audio segments corresponding to each speaker.
- **Pre-Trained Model**: Utilizes Hugging Face's `pyannote/speaker-diarization` for state-of-the-art diarization.
- **GPU Acceleration**: Supports CUDA for faster processing when a GPU is available.

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `pyannote.audio`
  - `pydub`
  - `huggingface_hub`

## Installation

1. Clone the repository:
   git clone https://github.com/AmirTahaMim/speaker-diarization.git
   cd speaker-diarization

2. Install the required Python packages:
   pip install torch pyannote.audio pydub huggingface_hub

3. Install the required audio codecs for `pydub`:
   - For Linux: Install `ffmpeg` via your package manager, e.g., `sudo apt install ffmpeg`.
   - For Windows/Mac: Download and install `ffmpeg` from [FFmpeg.org](https://ffmpeg.org/).

4. Authenticate with Hugging Face:
   from huggingface_hub import login
   login("your_huggingface_token")

## Usage

1. Place your audio file in the working directory. The file should be in `.wav` format.
2. Update the `audio_file` variable in the script with the path to your audio file.
3. Run the script:
   python diarization_splitter.py

4. The script will:
   - Perform speaker diarization on the input audio file.
   - Save individual `.wav` files for each detected speaker in the working directory.

## Output
For each detected speaker, the script generates a `.wav` file named `<SPEAKER_ID>.wav` containing all segments attributed to that speaker.

Example:
- `SPEAKER_0.wav`
- `SPEAKER_1.wav`

## Example Code

Here’s a snippet of the main pipeline:

from huggingface_hub import login
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch

# Authenticate and load the pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="your_token")
pipeline.to(device)

# Process the audio file
audio_file = "path_to_your_audio.wav"
diarization = pipeline(audio_file)

# Split audio into speaker-specific segments
audio = AudioSegment.from_wav(audio_file)
speaker_segments = {}

for segment, track, speaker in diarization.itertracks(yield_label=True):
    start_time = segment.start * 1000  # Convert to milliseconds
    end_time = segment.end * 1000
    speaker_audio = audio[start_time:end_time]
    if speaker not in speaker_segments:
        speaker_segments[speaker] = []
    speaker_segments[speaker].append(speaker_audio)

# Save each speaker's audio
for speaker, segments in speaker_segments.items():
    combined_audio = AudioSegment.empty()
    for segment in segments:
        combined_audio += segment
    combined_audio.export(f"{speaker}.wav", format="wav")

## Notes
- Ensure your Hugging Face token has appropriate permissions to access the pre-trained model.
- Audio files should ideally be in `.wav` format for compatibility with `pydub`.

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for the `pyannote.audio` library.
- [PyDub](https://github.com/jiaaro/pydub) for audio processing.

---

Happy coding! 🎧
