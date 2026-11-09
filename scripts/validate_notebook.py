"""Executa o notebook em um kernel limpo do Python atual e salva as saídas."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Exporta também uma cópia HTML.")
    parser.add_argument("--transport", choices=["tcp", "ipc"], default="tcp",
                        help="Transporte local do kernel; ipc dispensa portas TCP em sistemas compatíveis.")
    parser.add_argument("--in-process", action="store_true",
                        help="Executa em IPython no processo atual, sem iniciar um kernel externo.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "options_lab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)

    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        import os

        shell = InteractiveShell.instance()
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            count = 0
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                with capture_output() as captured:
                    result = shell.run_cell(cell.source, store_history=False)
                result.raise_error()
                cell.execution_count = count
                cell.outputs = [nbformat.v4.new_output(
                    "display_data", data=output.data, metadata=output.metadata,
                ) for output in captured.outputs]
                for name in ("stdout", "stderr"):
                    content = getattr(captured, name)
                    if content:
                        cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
        finally:
            os.chdir(previous_directory)
    else:
        execute_kernel(notebook, root, args.transport)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    mode = "IPython no processo" if args.in_process else "kernel Jupyter"
    print(f"Notebook validado: {count} células executadas sem erro ({mode}).")
    if args.html is not None:
        from nbconvert import HTMLExporter

        html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML: {args.html}")


def execute_kernel(notebook, root, transport):
    """Inicia um kernel do mesmo ambiente do script, sem instalação global."""
    with tempfile.TemporaryDirectory(prefix="options-lab-kernel-") as temporary:
        directory = Path(temporary) / "python3"
        directory.mkdir()
        (directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3 (ipykernel)", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
            transport=transport,
            ip=str(Path(temporary) / "kernel") if transport == "ipc" else "127.0.0.1",
        )
        client = NotebookClient(
            notebook, km=manager, timeout=600, record_timing=False,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute(cleanup_kc=True)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
