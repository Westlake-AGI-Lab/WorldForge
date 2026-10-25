# Prepared examples

Run the commands in [the quick start](../README.md#quick-start) to generate all three examples. The input image, warped RGB frames, and visibility masks are included; no depth estimation or ground truth is needed.

| Example | Input frames | H3 sampling frames |
| --- | ---: | ---: |
| `91d19c97__arc_right` | 49 | 56 |
| `8069d6f7__smooth` | 49 | 56 |
| `9d821bd4__orbit_return_push__90` | 90 | 90 |

The two shorter sequences are padded during encoding and trimmed on export. The longer example sets its frame count and aligned input framing in `cases.json`; both sampling presets reuse the same prepared cache. Masks are white in visible regions and black in holes.
