# Clean and normalise audio, and optionally add video background.

`python av.py [--action action] -i input_file -o output_file [-v background_video_file]`

- `input_file` can be any media file that ffmpeg can read (including a video file)
- If extensions are omitted for input they will be guessed (`.wav`, `.mp3`, `.mp4` for audio, `.mp4` for video)
- If extensions are omitted for output `.wav` or `.mp4` will be used.

## auto
action will be guessed from extension of output_file (`.wav` or `.mp3` -> audio, `.mp4` -> videopatch)

## audio 
read an audiofile, clean and normalise it, and save the output
`python av.py [--action audio] -i audio_file -o output_audiofile`

## video
read an audiofile, clean and normalise it, add a video (with looping if required), and save the output. 
`python av.py [--action video] -i audio_file -v video_file -o output_videofile`

If audio_file == video_file, this cleans the audio of an existing video

## Other options
|Option||
|-|-|
|`--no_enhance`|skip the audio enhance step|
|`--loudness`|target loudness (in dB) for normalisation (default `-20.0`)|
|`--root`|root directory (or shortcut) for `-i,` `-o`|
|`--vroot`|root directory (or shortcut) for `-v`|

## Shortcuts

Shortcuts to commonly used directories can be defined in `shortcuts.yaml` (see `shortcuts example.yaml`for details). 
They can be used with `--root` and `--vroot`.
