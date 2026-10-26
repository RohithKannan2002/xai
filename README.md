# XAI Practicals — Topic-wise folders

The practical code is split into separate folders at the repository root:

```text
xai/
├── tabular/
│   ├── main.py
│   └── requirements.txt
├── image/
│   ├── main.py
│   └── requirements.txt
├── text/
│   ├── main.py
│   └── requirements.txt
└── xai_practical_reference/   # topic-wise reference/source viewer package
```

## Install and run

Run commands from the repository root. Create/activate your virtual environment first.

**Tabular**
```powershell
python -m pip install -r tabular/requirements.txt
python tabular/main.py
```

**Image**
```powershell
python -m pip install -r image/requirements.txt
python image/main.py
```

**Text**
```powershell
python -m pip install -r text/requirements.txt
python text/main.py
```

The image script uses the sample path `/content/cat.jpg`; change it to a valid image path on your computer. Tabular and text practicals may download data or a pretrained model on first run. Install only the dependencies for the topic you need.
