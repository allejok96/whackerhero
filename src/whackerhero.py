#!/usr/bin/env python
import argparse
import os
import sys
from math import floor, ceil
from typing import NamedTuple

import numpy
from PIL import Image, ImageDraw, ImageFont
from mido import MidiFile
from moviepy.editor import AudioFileClip, VideoFileClip, VideoClip, CompositeVideoClip, CompositeAudioClip, ImageClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_resize
from proglog import TqdmProgressBarLogger

BACKGROUND_COLOR = (0, 0, 0)  # black
BOTTOM_MARGIN = 0.2  # of screen height
END_TIME = 4  # sec
FONTS = 'arial.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf'
FONT_HEIGHT = 0.4  # of bottom area
HIT_EFFECT_COLOR = (255, 255, 255)  # white
HIT_EFFECT_TIME = 1  # sec
HIT_LINE_COLOR = (255, 255, 255)  # white
LINE_OPACITY = 180  # 0 is transparent and 255 is opaque
LINE_THICKNESS = 0.05  # of column width
NOTE_WIDTH = 0.3  # of column width
SIDE_MARGIN = 0.05  # of screen width

# Names and colors of notes, should be in this order
KEYS = {
    'C': 'ff0000',
    'C#': 'fa6c20',
    'D': 'ff9300',
    'D#': 'ffc000',
    'E': 'fffe00',
    'F': '83ff00',
    'F#': '00ff74',
    'G': '00ffe5',
    'G#': '002bff',
    'A': '8000ff',
    'A#': 'ae5eff',
    'B': 'ff00ff'
}

OPAQUE = 255


def blend(color1: tuple, color2: tuple, ratio: float):
    """Blend two colors"""
    return tuple(map(lambda n1, n2: int(n1 + (n2 - n1) * ratio), color1, color2))


def clamp(n, upper):
    """Clamp a number between 0 and a upper limit"""
    return max(0, min(n, upper))


def rect(surface: numpy.ndarray, left, right, top, bottom, color: tuple):
    """Draw a rectangle, antialiased at top and bottom"""

    # Add alpha value
    if len(color) == 3:
        color = (*color, OPAQUE)

    # Keep all values within the surface bounds
    height, width, dimensions = surface.shape
    left = int(clamp(left, width - 1))
    right = int(clamp(right, width - 1))
    top = clamp(top, height - 1)
    bottom = clamp(bottom, height - 1)

    top_start = ceil(top)
    top_end = floor(top)
    bottom_start = floor(bottom)
    bottom_end = ceil(bottom)

    # Body
    surface[top_start:bottom_end, left:right] = color
    # Top edge alpha
    surface[top_start:top_end, left:right, 3] = int((top_start - top) * color[3])
    # Bottom edge alpha
    surface[bottom_start:bottom_end, left:right, 3] = int((bottom - bottom_start) * color[3])


def vline(surface: numpy.ndarray, center, top, bottom, thickness, color: tuple):
    return rect(surface, center - thickness / 2, center + thickness / 2, top, bottom, color)


def hline(surface: numpy.ndarray, center, left, right, thickness, color: tuple):
    return rect(surface, left, right, center - thickness / 2, center + thickness / 2, color)


class Note(NamedTuple):
    key: int
    start: int
    stop: int


class Painter:
    """MIDI to image generator"""

    def __init__(self, midifile: str, width: int, height: int, speed: int):
        self.width, self.height = width, height
        self.fall_time = speed

        # Load MIDI data
        midi = MidiFile(midifile)
        self.duration = midi.length
        total_sec = 0
        pressed_keys = {}  # {key: start_sec}
        self.notes: list[Note] = []

        # Iterates all midi messages in the order they were recorded
        for msg in midi:
            # Time is reported as seconds since last message
            total_sec += msg.time
            # Remember when a key is pressed down
            if msg.type == 'note_on':
                if msg.note not in pressed_keys:
                    pressed_keys[msg.note] = total_sec
            # Save a note when a key is released
            if msg.type == 'note_off':
                if msg.note in pressed_keys:
                    self.notes.append(Note(key=msg.note, start=pressed_keys[msg.note], stop=total_sec))
                    del pressed_keys[msg.note]

        # All keys that will be displayed
        self.used_keys = sorted(set(note.key for note in self.notes))

        # Split color codes in three parts, decode them as hex values and convert to something between 0-1
        self.colors = [tuple(int(xx, 16) for xx in (rgb[0:2], rgb[2:4], rgb[4:6])) for rgb in KEYS.values()]

        # Common measurements
        self.column_width = (width - width * SIDE_MARGIN * 2) / len(self.used_keys)
        self.note_width = self.column_width * NOTE_WIDTH
        self.line_thickness = max(1, int(self.column_width * LINE_THICKNESS))
        self.columns = [width * SIDE_MARGIN + (i + 0.5) * self.column_width for i in range(len(self.used_keys))]

        # Save static parts of the drawing
        self.background = self.draw_static()

        # The mask will be generated for each frame that is drawn
        # and moviepy will request the mask after each nomral frame
        # We cannot rely on this order of excecution though, so we save
        # the frame timestamp together with the mask to make sure they match
        self.mask_timestamp = -1.0
        self.mask = None

    def draw_static(self):
        """Generate a numpy image of all the stuff that doesn't move"""

        width = self.width
        height = self.height
        fadepoint = (1 - BOTTOM_MARGIN / 2) * height

        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)

        # Decrease font height if columns are too narrow
        font_size = int(min(self.column_width / 1.5, height * BOTTOM_MARGIN / 4))

        # Try to load fonts until it works
        for f in FONTS:
            try:
                font = ImageFont.truetype(f, size=font_size)
                break
            except OSError:
                continue
        else:
            # It didn't work, use the super small fallback font
            font = ImageFont.load_default()

        letters = [letter for letter in KEYS.keys()]

        for i, x in enumerate(self.columns):
            x = int(x)
            key = self.used_keys[i]
            color = self.colors[key % 12]
            letter = letters[key % 12]

            # Draw vertical key lines
            draw.line([(x, 0), (x, fadepoint - 1)], fill=(*color, LINE_OPACITY), width=self.line_thickness)

            # Draw text
            # Requires pillow 8.0.0 for correct anchor
            y = int((1 - BOTTOM_MARGIN / 4) * height)
            draw.text((x, y), text=letter, fill=color, anchor='mm', font=font)

        # Convert to numpy image
        array = numpy.array(image)

        # Draw horizontal hit-line
        hitline = (1 - BOTTOM_MARGIN) * height
        hline(array, hitline, 0, width, thickness=self.line_thickness, color=(*HIT_LINE_COLOR, LINE_OPACITY))

        return array

    def draw_notes(self, seconds):
        """Generate a numpy image (video frame) for a specified time"""

        width = self.width
        height = self.height

        surface = self.background.copy()

        # Bottom line
        hitline = (1 - BOTTOM_MARGIN) * height
        # Where note edge touches hit-line
        hitpoint = hitline - self.line_thickness / 2
        # Pixels per second
        pps = hitpoint / self.fall_time
        # Where note becomes invisible
        fadepoint = (1 - BOTTOM_MARGIN / 2) * height

        # Columns
        for i, x in enumerate(self.columns):
            key = self.used_keys[i]
            color = self.colors[key % 12]

            for note in self.notes:
                if note.key != key:
                    continue

                # Cut notes at the fade point
                bottom = min(fadepoint, note.start * -pps + seconds * pps)
                top = min(fadepoint, note.stop * -pps + seconds * pps)

                note_color = color

                # Progress of hit effect, 1 = just hit, 0 = invisible
                fadeout_stage = 1 - (bottom - hitpoint) / (HIT_EFFECT_TIME * pps)

                # Neither note, nor hit-line effect is visible
                if bottom < 0 or (top >= fadepoint and fadeout_stage < 0):
                    continue

                # Hit effect of note (color shift)
                if 0 < fadeout_stage < 1:
                    note_color = blend(color, HIT_EFFECT_COLOR, fadeout_stage)

                # Draw note
                vline(surface, x, top, bottom, self.note_width, color=(*note_color, OPAQUE))

                # Draw line effect
                if 0 < fadeout_stage < 1:
                    w = self.column_width - (self.column_width - self.note_width) * fadeout_stage ** 2
                    h = self.line_thickness + 2 * self.line_thickness * fadeout_stage ** 2
                    alpha = int(LINE_OPACITY + (OPAQUE - LINE_OPACITY) * fadeout_stage)
                    hline(surface, hitline, x - w / 2, x + w / 2, thickness=h, color=(*HIT_EFFECT_COLOR, alpha))

        # Extract the alpha layer (copied from VideoClip.to_mask)
        self.mask_timestamp = seconds
        self.mask = surface[:, :, 3] / 255

        # Make the mask fade out everything below the hit-line
        for line in range(int(hitline), int(fadepoint)):
            self.mask[line, :] *= 1 - (line - hitline) / (fadepoint - hitline)

        # Return the image without the alpha layer
        return surface[:, :, :3]

    def draw_mask(self, t):
        """Return mask if it has the correct timestamp, else re-generate it"""

        if self.mask_timestamp != t:
            print('re-drawing mask', t)
            self.draw_notes(t)

        return self.mask


class GooeyCompatibleParser(argparse.ArgumentParser):
    """An ArgumentParser that simply ignores Gooey stuff"""

    def add_argument(self, *args, widget=None, gooey_options=None, **kwargs):
        super().add_argument(*args, **kwargs)


class PercentageLogger(TqdmProgressBarLogger):
    """Prints percentage instead of a progress bar (needed by Gooey to read progress)"""

    def bars_callback(self, bar, attr, value, *args, **kwargs):
        print(f'{100 * value // self.bars[bar]["total"]}%')


def main(parser=None):
    parser = parser or GooeyCompatibleParser()
    parser.add_argument('-a', dest='audio', help='Audio track', widget='FileChooser')
    parser.add_argument('-i', dest='image', help='Background image', widget='FileChooser')
    parser.add_argument('-p', dest='test', help='Generate a preview', action='store_true')
    parser.add_argument('-s', dest='size', help='Video dimension (WIDTHxHEIGHT)', default='1280x720')
    parser.add_argument('-v', dest='video', help='Background video', widget='FileChooser')
    parser.add_argument('--fps', help='Frame rate', type=int, default=30, widget='IntegerField')
    parser.add_argument('--opacity', type=int, default=30, help='Background visibility (1-100)',
                        widget='Slider')
    parser.add_argument('--mute', help='Mute background video', action='store_true')
    parser.add_argument('--speed', help='Seconds from top to bottom', type=int, default=10,
                        widget='IntegerField')
    parser.add_argument('midi', help='Input MIDI file', widget='FileChooser')
    parser.add_argument('dest', help='Output video file', default='output.mp4', widget='FileSaver',
                        gooey_options={'validator': {'test': '"." in user_input',
                                                     'message': 'Missing file name extension'}})

    options = parser.parse_args()

    audio = None
    bg = None
    video_stack = []
    width, height = (int(n) for n in options.size.split('x'))

    # Load MIDI data
    painter = Painter(options.midi, width, height, speed=options.speed)
    duration = painter.duration + options.speed + END_TIME

    # Mix audio files
    # (will create a temp file in any case)
    if options.audio:
        if len(options.audio) == 1:
            audio = AudioFileClip(options.audio[0]).set_start(options.speed)
        else:
            audio = CompositeAudioClip([AudioFileClip(a).set_start(options.speed) for a in options.audio])
            pass
        duration = max(duration, audio.duration)

    # Prepare background video
    if options.video:
        bg = VideoFileClip(options.video)

        # Scale background video proportionally so it covers the whole screen
        # On-the-fly resizing with moviepy just locked up when I've tried it...
        # so here we use the external ffmpeg resizing, which creates a new video file
        if (bg.w != width and bg.h != height) or (bg.w < width or bg.h < height):
            prop = width / bg.w, height / bg.h
            w = bg.w * max(prop) // 2 * 2
            h = bg.h * max(prop) // 2 * 2
            # Check if we have done this already
            base, ext = os.path.splitext(bg.filename)
            resized = f'{base}_{w}x{h}{ext}'
            if not os.path.exists(resized):
                # Scale video and save it for future (re)use
                ffmpeg_resize(bg.filename, resized, (w, h))

            bg = VideoFileClip(resized)

        if options.mute:
            bg = bg.set_audio(None)

        bg = bg.set_start(options.speed)
        duration = max(duration, bg.duration)

    # Add background image
    elif options.image:
        video_stack.append(ImageClip(options.image, duration=duration))

    if bg:
        bg = bg.set_position(('center', 'center')).set_opacity(options.opacity / 100)
        video_stack.append(bg)

    # Create falling notes
    notes = VideoClip(painter.draw_notes, duration=duration).set_audio(audio)
    mask = VideoClip(painter.draw_mask, duration=duration, ismask=True)
    notes = notes.set_mask(mask)
    video_stack.append(notes)

    # Mix all layers
    video = CompositeVideoClip(video_stack, size=(width, height), bg_color=BACKGROUND_COLOR) \
        .set_duration(duration).fadein(1).fadeout(1)

    if options.test:
        video = video.subclip(options.speed + 10, options.speed + 20)

    # Gooey cannot read a normal progress bar, so we print percentage if running non-interactively
    logger = TqdmProgressBarLogger() if sys.stdout.isatty() else PercentageLogger()

    # Start rendering
    video.write_videofile(options.dest, fps=options.fps, logger=logger)


def gui():
    """Entry point when running the executable - try to load GUI if no arguments are specified"""

    if len(sys.argv) > 1:
        main()
        return

    try:
        from gooey import Gooey, GooeyParser

        # This calls the main function through the Gooey decorator, same as:
        # @Gooey
        # def main(): ...
        # main()
        Gooey(
            program_name='Whacker hero',
            program_description='Generate play-along videos for Boomwhackers',
            progress_regex=r'^(\d+)%',
            hide_progress_msg=True,
            suppress_gooey_flag=True,
        )(main)(parser=GooeyParser())

    except ImportError:
        print('Run `pip install Gooey` to install graphical user interface, or try `whackerhero --help`')


if __name__ == '__main__':
    main()
