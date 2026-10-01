# Windows

The app is written and tested on macOS. The code is prepared for Windows (see
*What differs* below), and this page is the checklist for running it there,
building the Windows download and publishing it.

## Run from source

You need **Python 3.11–3.13 from python.org** (x64; not the Microsoft Store
build) and about 6 GB of free disk.

```powershell
winget install Python.Python.3.13
git clone https://github.com/EfeUtku0/BiSn-Particle-SEM-Analyzer.git
cd BiSn-Particle-SEM-Analyzer
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app\gui.py
```

Afterwards `run.bat` starts it with a double-click. The first analysis downloads
the Cellpose (`cpsam`, ~1.2 GB) and EasyOCR weights into your home folder, once.

`pip` installs the CPU build of PyTorch on Windows. With an NVIDIA card, the
CUDA build makes Cellpose many times faster:
`.venv\Scripts\pip install --force-reinstall --no-deps torch==<the version pip installed> torchvision==<same> --index-url https://download.pytorch.org/whl/cu126`
(the app picks the GPU up by itself). Pin the versions `pip list` shows: an
index that lacks them (cu124 no longer carries torch 2.14) quietly installs an
older torch instead. Measured on an RTX 3050 Laptop (4 GB): the first analysis
68 s, every one after it 15 s, against 8 to 9 min per image on that laptop's CPU.

Build the download from a separate CPU-only environment (`py -3.13 -m venv
.venv-cpu`, then the same `pip install -r requirements.txt`): the CUDA build
of torch alone adds about 2.5 GB, which would push the zip past GitHub's
2 GB limit for a release asset.

## Checklist

Run from source first; the console shows any error.

1. The window opens at a sensible size; text is not cramped or oversized
   (the UI is sized in pixels, 13 px base).
2. The eight example micrographs are in the library on a first launch.
3. **Analyze** one example. Note the time of the first and of the second
   analysis (CPU-only PyTorch is the slow case to measure).
4. The scale bar is read (the chip next to the image), and the results panel
   and charts fill in.
5. **Import**: the button offers *Images…* and *A whole folder…*; dragging a
   folder or files onto the window works too.
6. Shortcuts: `Ctrl+Z` undo, `Ctrl+0` fit, `Ctrl`+scroll zoom, number keys,
   `Space`, `Delete` on a library row, `F2` rename. The Guide says `Ctrl`, not ⌘.
7. **Save** an export (PNG): the file opens and the chart text uses Segoe UI.
8. Close the window during nothing, then reopen: the library and the analyses
   are back (`%APPDATA%\SEM Particle Analyzer`).
9. A folder or file name with Turkish letters (ö, ş, ı, İ) imports and analyzes.

## Build the download

PyInstaller does not cross-compile, so this runs on Windows:

```powershell
.venv-cpu\Scripts\pyinstaller --noconfirm --clean sem_analyzer.spec
```

(Plain `.venv` is fine too if you never installed the CUDA build into it.)

Run one analysis from source before building: the spec copies the model
weights from `%USERPROFILE%\.cellpose` and `%USERPROFILE%\.EasyOCR`, and a build
without them would download 1.2 GB on the user's first analysis.

Check the build without opening a window (the windowed build has no console,
so the result goes to the log):

```powershell
$env:SEMPA_SELFTEST = "$PWD\sample_images\example-5-mixed.jpeg"
$env:SEMPA_SELFTEST_OUT = "$env:TEMP\selftest.png"
& "dist\SEM Particle Analyzer\SEM Particle Analyzer.exe"
Get-Content "$env:APPDATA\SEM Particle Analyzer\sem.log"
Remove-Item Env:SEMPA_SELFTEST, Env:SEMPA_SELFTEST_OUT
```

It should end with `SELFTEST OK`; on the Mac the same image gives
469 particles, mean 832 nm, 227 solid (CPU results may differ by a few
particles). Then zip the whole `dist\SEM Particle Analyzer\` folder:

```powershell
Compress-Archive "dist\SEM Particle Analyzer" SEM-Particle-Analyzer-windows.zip
```

A GitHub release asset may be up to 2 GB.

## Installing the download

Unzip anywhere and start `SEM Particle Analyzer.exe`. The app is unsigned, so
the first start shows *Windows protected your PC*: **More info → Run anyway**.

## What differs from macOS

| | macOS | Windows |
|---|---|---|
| Closing the window | hides it, the Dock icon brings it back, ⌘Q quits | quits (after saving) |
| Import button | one Finder panel for files and folders | asks *Images…* or *A whole folder…* |
| Shortcut names in the UI | ⌘ | Ctrl |
| Exported PNGs | tagged Display P3 | untagged (sRGB) |
| Chart face for UI-like parts | baked from the system font (SF) | Segoe UI |
| Acceleration | Apple GPU (MPS) | NVIDIA (CUDA) if installed, else CPU |
| Data folder | `~/Library/Application Support/SEM Particle Analyzer` | `%APPDATA%\SEM Particle Analyzer` |
| Training / golden folders | Desktop | Documents |
| Log (windowed build) | none | `sem.log` in the data folder |
