# SEM Particle Analyzer

Measures particle sizes in SEM micrographs and classifies each particle as
undercooled (liquid) or solid — and, for solid ones, its pattern (Janus, Stripe,
Composite, Lamellar). Scale is read from the image's own scale bar, so no manual
calibration is needed.

Runs fully offline: the segmentation and OCR models ship inside the app.

---

## Installing (for someone who just wants to use it)

### macOS

1. Copy **SEM Particle Analyzer.app** into your `Applications` folder.
2. **The first launch needs one extra step**, because the app isn't signed with
   an Apple developer certificate: **right-click the app → Open → Open**.
   Double-clicking the first time shows "cannot be opened" — that's Gatekeeper,
   not a broken app. You only do this once.

   If macOS still refuses ("app is damaged"), open Terminal and run:

   ```bash
   xattr -cr "/Applications/SEM Particle Analyzer.app"
   ```

3. The first analysis takes a minute or so while the model loads. After that it
   is fast (segmentation results are cached).

### Windows

1. Unzip the **SEM Particle Analyzer** folder anywhere (e.g. Documents).
2. Run **SEM Particle Analyzer.exe** inside it.
3. SmartScreen may warn about an unknown publisher: **More info → Run anyway**.

Keep the whole folder together — the .exe needs the files beside it.

---

## Where your data is kept

Your image library (the folders and photos in the left panel), analyses and any
retrained model live **outside** the app, so updating or replacing the app never
touches them:

| Platform | Folder |
|---|---|
| macOS | `~/Library/Application Support/SEM Particle Analyzer` |
| Windows | `%APPDATA%\SEM Particle Analyzer` |
| Linux | `~/.local/share/SEM Particle Analyzer` |

Inside it:

- `library.json` — the left panel's folder tree
- `library.json.bak`, `library.startup.json`, `library.shrink.json` — automatic
  backups (see below)
- `session.pkl` (+ `.bak`) — your analyses
- `mask_cache/` — cached segmentations (safe to delete; only costs time)

**Your photos themselves are never moved, renamed or deleted.** The library only
stores paths to them. Folders in the app are virtual: dragging a photo between
them changes nothing on disk.

### If you delete something by accident

Removing rows is always deliberate (right-click → Remove, or the Delete key, with
a confirmation for folders). Even so, three rolling backups are kept:

- `library.json.bak` — the state before the most recent change
- `library.startup.json` — how things looked when you opened the app today
- `library.shrink.json` — the last state before the number of photos went *down*

To restore: quit the app, rename the backup you want to `library.json`, reopen.

---

## Training mode (correcting the model)

The 🎓 **Training** button lets you click particles to give them the right class,
then retrain the pattern model on what you marked.

This is **per-machine**: your labels go to `SEM Eğitim` in your Desktop
(macOS) or Documents (Windows/Linux), and a model you train is written to your
own data folder. Nobody else's copy is affected.

⚠️ **A retrained model overrides the one shipped in the app.** The bundled model
was trained on thousands of labelled particles across 32 photos; if you retrain
on just a handful of your own, results will get *worse*, and it won't be obvious
why. Only retrain once you have labelled a substantial set — and to go back to
the shipped model, delete `patternnet.pt` from the data folder above.

## Running from source (developers)

Python **3.11 or 3.12** recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python app/gui.py                 # or double-click run.command / run.bat
```

## Building a distributable app

Build on the platform you're targeting (PyInstaller does not cross-compile):

```bash
.venv/bin/pyinstaller --noconfirm --clean sem_analyzer.spec
```

- macOS → `dist/SEM Particle Analyzer.app`
- Windows → `dist/SEM Particle Analyzer/` (zip this folder to share it)

The build embeds the Cellpose and EasyOCR weights from your `~/.cellpose` and
`~/.EasyOCR` folders. If the spec prints a `WARNING: ... weights not found`,
run one analysis from source first so those models get downloaded, then rebuild —
otherwise the packaged app will need internet on its first run.

To verify a build without a GUI:

```bash
SEMPA_SELFTEST=/path/to/image.jpeg "dist/SEM Particle Analyzer.app/Contents/MacOS/SEM Particle Analyzer"
```
