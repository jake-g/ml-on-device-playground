"""Motion Synth: an IMU motion-driven synthesizer for the XIAO nRF52840 Sense.

Move, lift, toss, or drop the board on a USB-C cable: picking it up raises
a whoosh, while it is weightless the host plays a whistle whose pitch
follows ballistic height (rising to the apex, then winding down as it falls),
the landing spike triggers an impact sound (`boom`, `squish`, `squelch`,
`screech`, `zap`, `boing`, `shatter`, `clap`, or `beep`) sized by impact and
airtime, and each bounce replays a smaller hop. Everything is plain
subtractive synthesis. The `Arduino/MotionSynth` sketch adds gyro lines that
keep the lift detector's sense of up steady.

Modules:
  detector: device spec, physics, flight detector, thresholds learned
    from `Python/data` plus saved flights and handling, and simulations.
  synth: subtractive synth, the engine that plays flight events and
    records clips and handling, and speaker or WAV output.
  widget: PyQt5 tab for the playground UI and the standalone window.
  __main__: CLI (`python -m motion_synth --help`).
"""

__version__ = "0.2.0"
