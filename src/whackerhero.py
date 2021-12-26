#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from collections import namedtuple
from math import ceil

import numpy as np
from mido import MidiFile
from moviepy.editor import AudioFileClip, VideoFileClip, VideoClip, CompositeVideoClip, ImageClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_resize

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


class Color(namedtuple('Color', 'r g b a')):
    a: int

    def __new__(cls, r, g, b, a=OPAQUE) -> Color:
        """Create a new color and make sure all values are int"""
        return super().__new__(cls, int(r), int(g), int(b), int(a))

    @classmethod
    def fromhex(cls, string: str):
        """Create a color from a hexadecimal string"""
        return cls(int(string[0:2], 16), int(string[2:4], 16), int(string[4:6], 16))

    def opacity(self, a):
        """Return a copy of this color with a different opacity"""
        return Color(self.r, self.g, self.b, int(a))

    def blend(self, other):
        """Blending another color on top of this"""

        a1 = self.a / OPAQUE
        a2 = other.a / OPAQUE
        # The relationship between a1 and a2; 1:1 = 0.5; 1:0 = 0
        median_opacity = (a2 + 1 - a1) / 2
        rbg = (self[i] + (other[i] - self[i]) * median_opacity for i in range(3))
        return Color(*rbg, min(self.a + other.a, OPAQUE))


# Represents a MIDI note
Note = namedtuple('Note', 'key start stop')


class AntialiasedDraw:
    """Draws on numpy images, similar to ImageDraw, but with antialiasing and alpha composite"""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def text(self, xy, text, fill=None, font=None, size=10):
        """Draw centered text using Pillow"""

        from PIL import Image, ImageDraw, ImageFont

        # Try to load fonts until it works
        fonts = [font] if font else FONTS
        for f in fonts:
            try:
                loaded_font = ImageFont.truetype(f, size=size)
                break
            except OSError:
                continue
        else:
            print('Cannot find font, falling back to microscopic default font')
            loaded_font = ImageFont.load_default()

        image = Image.fromarray(self.arr, "RGBA")
        # Requires pillow 8.0.0 for correct anchor
        ImageDraw.Draw(image).text([int(n) for n in xy], text, fill, loaded_font, anchor='mm')
        self.arr[...] = np.array(image)

    def vline(self, center, top, bottom, width, color: Color):
        """Vertical line"""
        self.rectangle(center - width / 2, top, center + width / 2, bottom, color)

    def hline(self, center, left, right, width, color: Color):
        """Horizontal line"""
        self.rectangle(left, center - width / 2, right, center + width / 2, color)

    def box(self, x, y, width, height, color: Color):
        """Centered rectangle"""
        self.rectangle(x - width / 2, y - height / 2, x + width / 2, y + height / 2, color)

    def rectangle(self, left, top, right, bottom, color: Color):
        """Draw rectangle that is anti-aliased at top and bottom"""

        # Since left/right isn't anti-aliased, if it's thinner than 1px it will disappear
        # In that case, we clamp it to 1 px and lower the opacity
        if (right - left) < 1:
            color = color.opacity(min(1, right - left) * color.a)
            right = left + 1

        # If coordinates are negative they will mess up array[indexes]
        top = max(0, top)
        bottom = max(0, bottom)
        left = max(0, int(left))
        right = max(0, int(right))

        # Rounding up/down
        outer_top = int(top)
        inner_top = int(ceil(top))
        inner_bottom = int(bottom)
        outer_bottom = int(ceil(bottom))

        if outer_top == outer_bottom:
            return

        top_alpha = 1 - (top - outer_top)
        bottom_alpha = 1 - (outer_bottom - bottom)

        # Draw body
        if inner_bottom - inner_top > 0:
            self.over(inner_top, inner_bottom, left, right, color)
        # Draw top/bottom aliasing
        if outer_bottom - outer_top > 1:
            self.over(outer_top, inner_top, left, right, color.opacity(top_alpha * color.a))
            self.over(inner_bottom, outer_bottom, left, right, color.opacity(bottom_alpha * color.a))
        # Draw only a 1 px line
        else:
            self.over(outer_top, outer_bottom, left, right, color.opacity(min(1, top_alpha + bottom_alpha) * color.a))

    def over(self, top, bottom, left, right, color):
        """Draw a transparent color over a part of the image"""

        # Don't do any compositioning when there is no transparency
        if color.a == OPAQUE:
            self.arr[top:bottom, left:right] = color
            return

        c1 = np.array(color[:3], np.uint8)
        c2 = self.arr[top:bottom, left:right, :3]

        # Normalise alpha to range 0..1
        a1 = color[3] / 255.0
        a2 = self.arr[top:bottom, left:right, 3] / 255.0

        # Alpha compositing equation, found on wikipedia
        ao = a1 + a2 * (1 - a1)
        co = (c1 * a1 + c2 * a2[..., np.newaxis] * (1 - a1)) / ao[..., np.newaxis]

        # Merge RGB and alpha (scaled back up to 0..255) back into single image
        self.arr[top:bottom, left:right] = np.dstack((co, ao * 255))

    def flatten(self):
        """Replace alpha channel with black"""
        for c in range(3):
            self.arr[:, :, c] -= np.minimum(255 - self.arr[:, :, 3], self.arr[:, :, c])


class Painter:
    """MIDI to image generator"""

    def __init__(self, midifile: str, width: int, height: int, speed: int, show_text: bool, masked: bool, font: str):
        self.width, self.height = width, height
        self.fall_time = speed
        self.show_text = show_text
        self.masked = masked
        self.font = font

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

        self.colors = [Color.fromhex(colorcode) for colorcode in KEYS.values()]

        # Common measurements
        self.column_width = (width - width * SIDE_MARGIN * 2) / len(self.used_keys)
        self.note_width = self.column_width * NOTE_WIDTH
        self.line_thickness = self.column_width * LINE_THICKNESS
        self.columns = [width * SIDE_MARGIN + (i + 0.5) * self.column_width for i in range(len(self.used_keys))]

        # Save static parts of the drawing
        self.background = self.draw_static()

        # The mask will be generated for each frame that is drawn
        # and moviepy will request the mask after each normal frame
        # We cannot rely on this order of execution though, so we save
        # the frame timestamp together with the mask to make sure they match
        self.mask_timestamp = -1.0
        self.mask = None

    def draw_static(self):
        """Generate a image of all the stuff that doesn't move"""

        width = self.width
        height = self.height
        fadepoint = (1 - BOTTOM_MARGIN / 2) * height

        arr = np.zeros((height, width, 4), np.uint8)
        draw = AntialiasedDraw(arr)

        # Decrease font height if columns are too narrow
        font_size = int(min(self.column_width / 1.5, height * BOTTOM_MARGIN / 4))

        letters = [letter for letter in KEYS.keys()]

        for i, x in enumerate(self.columns):
            key = self.used_keys[i]
            color = self.colors[key % 12]
            letter = letters[key % 12]

            # Draw vertical key lines
            draw.vline(x, 0, fadepoint - 1, self.line_thickness, color.opacity(LINE_OPACITY))

            # Draw text
            if self.show_text:
                y = (1 - BOTTOM_MARGIN / 4) * height
                draw.text((x, y), text=letter, fill=color, font=self.font, size=font_size)

        # Draw horizontal hit-line
        hitline = (1 - BOTTOM_MARGIN) * height
        color = Color(*HIT_LINE_COLOR).opacity(LINE_OPACITY)
        draw.hline(hitline, 0, width, self.line_thickness, color)

        return arr

    def draw_notes(self, seconds):
        """Generate a image for a specified time (video frame)"""

        # Print progress percentage if running in Gooey
        if not sys.stdout.isatty():
            print(int(seconds / self.duration * 100), flush=True)

        width = self.width
        height = self.height

        arr = self.background.copy()
        draw = AntialiasedDraw(arr)

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

                bottom = note.start * -pps + seconds * pps
                top = note.stop * -pps + seconds * pps
                note_color = color

                # Progress of hit effect, 1 = just hit, 0 = invisible
                fadeout_stage = 1 - (bottom - hitpoint) / (HIT_EFFECT_TIME * pps)

                # Hit effect of note (color shift)
                if 0 < fadeout_stage < 1:
                    note_color = color.blend(Color(*HIT_EFFECT_COLOR, fadeout_stage * OPAQUE))

                # Draw note (cut it at after fadeout)
                if bottom > 0 and top < fadepoint:
                    draw.vline(x, top, min(fadepoint, bottom), self.note_width, note_color)

                # Draw line effect
                if 0 < fadeout_stage < 1:
                    w = self.column_width - (self.column_width - self.note_width) * fadeout_stage ** 2
                    h = self.line_thickness + 2 * self.line_thickness * fadeout_stage ** 2
                    draw.box(x, hitline, w, h, Color(*HIT_LINE_COLOR).opacity(OPAQUE * fadeout_stage ** 2))

        hitline = int(hitline)
        fadepoint = int(fadepoint)

        # Make everything fade out below the hit-line
        for line in range(hitline, fadepoint):
            alpha = int(255 * (line - hitline) / (fadepoint - hitline))
            arr[line, :, 3] -= np.minimum(arr[line, :, 3], alpha)

        if self.masked:
            # Extract the alpha layer (copied from VideoClip.to_mask)
            self.mask_timestamp = seconds
            self.mask = arr[:, :, 3] / 255
        else:
            # Flatten image onto colored background, much faster than masking in moviepy
            draw.flatten()

        # Return the image without the alpha layer
        return arr[:, :, :3]

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


def main(parser=None):
    parser = parser or GooeyCompatibleParser()
    parser.add_argument('-a', dest='audio', help='Audio track', widget='FileChooser')
    parser.add_argument('-i', dest='image', help='Background image', widget='FileChooser')
    parser.add_argument('-p', dest='test', help='Generate a preview', action='store_true')
    parser.add_argument('-s', dest='size', help='Video dimension (WIDTHxHEIGHT)', default='1280x720')
    parser.add_argument('-v', dest='video', help='Background video', widget='FileChooser')
    parser.add_argument('--font', help='Custom font')
    parser.add_argument('--fps', help='Frame rate', type=int, default=30, widget='IntegerField')
    parser.add_argument('--opacity', type=int, default=30, help='Background visibility (1-100)',
                        widget='Slider')
    parser.add_argument('--mute', help='Mute background video', action='store_true')
    parser.add_argument('--no-text', help='Hide note names', action='store_true')
    parser.add_argument('--speed', help='Seconds from top to bottom', type=int, default=10,
                        widget='IntegerField')
    parser.add_argument('midi', help='Input MIDI file', widget='FileChooser')
    parser.add_argument('dest', help='Output video file', default='output.mp4', widget='FileSaver',
                        gooey_options={'validator': {'test': '"." in user_input',
                                                     'message': 'Missing file name extension'}})

    options = parser.parse_args()

    audio = None
    bg = None
    masked = any((options.video, options.image))
    video_stack = []
    width, height = (int(n) for n in options.size.split('x'))

    # Load MIDI data
    painter = Painter(options.midi, width, height, speed=options.speed, show_text=not options.no_text, masked=masked,
                      font=options.font)
    duration = painter.duration + options.speed + END_TIME

    # Prepare audio (will create a temp file)
    if options.audio:
        print('Preparing audio', flush=True)
        audio = AudioFileClip(options.audio).set_start(options.speed)
        duration = max(duration, audio.duration)

    # Prepare background video
    if options.video:
        bg = VideoFileClip(options.video)
        bgw, bgh = (bg.w, bg.h) if bg.rotation % 180 == 0 else (bg.h, bg.w)
        # Scale background video proportionally so it covers the whole screen
        # On-the-fly resizing with moviepy just locked up when I've tried it...
        # so here we use the external ffmpeg resizing, which creates a new video file
        if (bgw != width and bgh != height) or (bgw < width or bgh < height):
            prop = max(width / bgw, height / bgh)
            w = bgw * prop // 2 * 2
            h = bgh * prop // 2 * 2
            # Check if we have done this already
            base, ext = os.path.splitext(bg.filename)
            resized = f'{base}_{w}x{h}{ext}'
            if not os.path.exists(resized):
                # Scale video and save it for future (re)use
                print('Resizing video', flush=True)
                ffmpeg_resize(bg.filename, resized, (w, h))

            bg = VideoFileClip(resized)

        if options.mute:
            bg = bg.set_audio(None)

        bg = bg.set_start(options.speed)
        duration = max(duration, bg.duration)

    # Add background image
    elif options.image:
        bg = ImageClip(options.image, duration=duration)

    if bg:
        bg = bg.set_position(('center', 'center')).set_opacity(options.opacity / 100)
        video_stack.append(bg)

    # Create falling notes
    notes = VideoClip(painter.draw_notes, duration=duration).set_audio(audio)
    if masked:
        mask = VideoClip(painter.draw_mask, duration=duration, ismask=True)
        notes = notes.set_mask(mask)
    video_stack.append(notes)

    # Mix all layers
    video = CompositeVideoClip(video_stack, size=(width, height)).set_duration(duration).fadein(1).fadeout(1)

    if options.test:
        video = video.subclip(options.speed + 10, options.speed + 20)

    # When running in Gooey, disable pretty progress bar
    logger = 'bar' if sys.stdout.isatty() else None

    # Start rendering
    print('Rendering frames', flush=True)
    print(options.dest)
    if options.dest.endswith('.gif'):
        video.write_gif(options.dest, fps=options.fps, logger=logger)
    else:
        video.write_videofile(options.dest, fps=options.fps, logger=logger)



def gui():
    """Entry point for executable - load GUI if no arguments are specified"""

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
            # FIXME this function cannot be run from an interactive shell
            # When imported as a module, Gooey will run __main__.py in the WORKING DIRECTORY
            # When running as a script, Gooey will run `python sys.argv[0]`
            # On Windows, sys.argv[0] is perhaps an exe so that doesn't work.
            # Therefore we must call it directly.
            target=sys.argv[0],
            program_name='Whacker Hero',
            program_description='Boomwhacker play-along generator',
            progress_regex=r'^(\d+)$',
            hide_progress_msg=True,
            suppress_gooey_flag=True,
            clear_before_run=True,
        )(main)(parser=GooeyParser())

    except ImportError:
        print('Run `pip install Gooey` to enable graphical user interface, or try `whackerhero --help`')


if __name__ == '__main__':
    main()
