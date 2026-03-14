"""
This module defines the entry point for the command-line interface (CLI)
of the LAMPE project using Typer.
It allows users to interact with the application via the command line,
providing options for specifying the architecture and the path to a MAT file.
"""

import typer

app = typer.Typer()

@app.command()
def train(architecture: str, matpath: str):
    print(f"Architecture: {architecture}")
    print(f"MAT file path: {matpath}")

@app.command()
def inference():
    print("Running inference...")
