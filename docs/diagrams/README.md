# Diagrams

| File | What it is |
| --- | --- |
| `architecture.drawio` | Editable draw.io source: how the bot, the NAS, the Obsidian vault and Telegram fit together |
| `draw-day.drawio` | Editable draw.io source: what happens on a draw day |
| `*-light.svg`, `*-dark.svg` | The images shown in the main README (GitHub picks the one matching your theme) |
| `build_diagrams.py` | Generates all of the above from one layout |
| `icons/` | Official brand logos from [Simple Icons](https://simpleicons.org) (CC0), drawn in each brand's colour |

## Edit a diagram

Open a `.drawio` file in [diagrams.net](https://app.diagrams.net) (File, Open from, Device) or in VS Code
with the Draw.io Integration extension. You can also open it straight from GitHub:

* [Open architecture in draw.io](https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2Fgcjk768%2FHuat-Bot%2Fmain%2Fdocs%2Fdiagrams%2Farchitecture.drawio)
* [Open draw day in draw.io](https://app.diagrams.net/#Uhttps%3A%2F%2Fraw.githubusercontent.com%2Fgcjk768%2FHuat-Bot%2Fmain%2Fdocs%2Fdiagrams%2Fdraw-day.drawio)

The `.drawio` files embed the logos, so they work offline.

## Regenerate

```bash
python docs/diagrams/build_diagrams.py
```

Only the Python standard library is needed. Change the layout in `build_diagrams.py` and rerun it.
That keeps the light SVG, the dark SVG and the draw.io source in step.

## Logos

Docker, Telegram, Obsidian, Python, pandas, NumPy and Claude logos come from Simple Icons
16.33.0, which collects each brand's official mark. The SVG data is CC0 (see
`icons/SIMPLE_ICONS_LICENSE.md`). The trademarks still belong to their owners and are used here only
to name the tools the bot works with.
