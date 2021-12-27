# Whacker Hero

Create play-along videos for [Boomwhackers](https://boomwhackers.com/).

![example animation of falling notes](https://user-images.githubusercontent.com/7693838/147457241-99963c9e-357b-468e-b7dd-2ec2b31da3c4.gif)

## Installation

    pip install "whackerhero[gui]"

`[gui]` is optional. It installs `gooey` which may take a while on Linux.
You have to do a pip-install in order to use the gui, it cannot be launched directly from python or the shell.

Other dependencies are `mido` for midi reading, `numpy` for graphics, and `moviepy` for video rendering.

## Usage

![screenshot of config window](https://user-images.githubusercontent.com/7693838/147457255-23bd71e7-ba8b-4ef2-99df-f19f9c7179ba.jpg)

If you've installed the gui, it's pretty straightforward.
The most work goes into writing a music arrangement suitable for playing along to (see the example below).
There are many midi files on the internet, but they can be big and messy.
You should trim it down to only one or two octaves.

The midi file will *only* be used for visuals. If you want it for audio too, you have to convert it manually.

There is also a command line variant:

    whackercmd -a music.mp3 -i background.jpg midifile.mid output.mp4

You can even create gif images for some reason.

    whackercmd -s 400x300 --fps 15 midifile.mid output.gif

Due to the single-core design of moviepy, rendering takes forever.
Use the `test` option to generate a 10 sec long video before you do your final rendering.

### Adding play-along to a music video

1. Download the music video and [extract](https://www.audacityteam.org/) the music.
1. Import the music into your [DAW](https://lmms.io) of choice. Don't add any silence at the beginning.
1. Write notes in sync with the music. What you hear is what you'll see.
   If you make all notes the same short length, things will look less cluttered in the final video.
1. Export to a midi file.
1. Run `whackerhero`
1. Select the midi file and the music video.
1. Check the `test` box and press start.
1. If everything looks good, uncheck `test` and run again.
