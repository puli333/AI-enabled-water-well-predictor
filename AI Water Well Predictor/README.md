# AI Water Well Predictor

End-to-end demo with a Flask backend and a simple HTML/Leaflet frontend to estimate water well suitability, depth, discharge, and quality index using a synthetic ML model.

## Quickstart

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) (Optional) Parse CGWB dataset PDF to CSV

```bash
python data_ingestion.py
```

This downloads the CGWB PDF and extracts tables into `data/cgwb_tables.csv`. The backend will automatically use it if present; otherwise, it falls back to synthetic data.

4) Run the backend

```bash
python app.py
```

The API starts on `http://127.0.0.1:8000` with:
- `POST /predict`
- `GET /health`

5) Open the frontend

Open `index.html` in your browser (double-click). Click the map to set a point, fill the form, then press Predict.

If your browser blocks local file XHR, you can serve the folder:

```bash
python -m http.server 8080
```

Then browse to `http://127.0.0.1:8080/index.html`.

## Notes

- Models are trained at startup on synthetic data for demo purposes.
- CORS is enabled for all origins for simplicity.
- Adjust features and UI as needed to match real data.

## Anaconda users

If you have Anaconda installed but `conda` is not available in PowerShell, use the full Python path, install requirements, then run:

```powershell
"C:\Users\<YOU>\anaconda3\python.exe" -m pip install -r requirements.txt
"C:\Users\<YOU>\anaconda3\python.exe" data_ingestion.py
"C:\Users\<YOU>\anaconda3\python.exe" app.py
```


