if [ -d ".venv" ]; then
    echo "Directory exists."
else
    echo "Directory does not exist."
    uv venv
fi
    source .venv/bin/activate
    uv pip install -r requirement.txt
    jupyter lab
