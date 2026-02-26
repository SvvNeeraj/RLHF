$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
streamlit run streamlit_app.py
