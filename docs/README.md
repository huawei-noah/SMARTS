# Building Docs Locally

Execute the following to build the docs locally.
```bash
cd <path/to/SMARTS>
pip install -e .[doc]
make docs
python -m http.server -d docs/_build/html
# Open http://localhost:8000 in your browser
```