import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import sys
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return mo, os, project_root, sys


@app.cell
def _():
    from src.file_utils import extract_text_from_xml


    transkribus_text = extract_text_from_xml("data/reichenau_10_test/ground_truth/7474185.xml")
    print("\n".join(transkribus_text))
    return extract_text_from_xml, transkribus_text


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
