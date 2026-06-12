"""Walks up the folder path until an `__init__.py` chain is broken. Then runs the python file as a module with `py -m package_name.folder_name(s).filename`

You don't need an `__init__.py` to begin with, but the moment the first `__init__.py` is seen, the next time it's not found will end the search. This
should mirror how most packages look, without searching for infinity or up to root.

Stops at a depth of 10.
"""

import os
import runpy
import subprocess
import sys
import warnings
from pathlib import Path


def module_info_from_file(file_path: Path) -> tuple[str, Path]:
    file_path = file_path.resolve()

    if file_path.name == "__init__.py":
        raise SystemExit("Run a normal module file, not __init__.py itself.")

    if file_path.suffix != ".py":
        raise SystemExit(f"Not a Python file: {file_path}")

    parts = [file_path.stem]
    current = file_path.parent
    seen_init = False

    #TODO could make it check up to root instead of stop at depth of 10.
    #? We would just need to pass $workspacefolder to the script. It's not meant for manual use anyway
    i = 0
    while i < 10:
        # Stop once we've already entered a package chain and the next parent
        # is no longer part of it.
        i += 1
        if seen_init and not (current / "__init__.py").exists():
            break

        parts.insert(0, current.name)

        if (current / "__init__.py").exists():
            seen_init = True

        current = current.parent

        if current == current.parent:
            break
        
    if i == 10:
        raise SystemExit(f"Maximum search depth of {i} reached. Please make reasonable packages you psychopath")
    if not seen_init:
        raise SystemExit(f"{file_path} is not inside a package chain with __init__.py files.")

    # current is now the directory above the topmost package dir
    if len(parts) == 1:
        raise SystemExit(
            f"{file_path} is not inside a package chain with __init__.py files."
        )

    module_name = ".".join(parts)
    package_root = current
    return module_name, package_root


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: py tools/run_as_package.py path/to/file.py [args...]")

    # --- CAFFEINATE INTEGRATION ---
    # On macOS, prevent system sleep while this process is running.
    # The '-w <pid>' flag binds the caffeinate command to this script's lifecycle.
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["caffeinate", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            pass  # caffeinate not found on the system, fail silently
    # ------------------------------

    target = Path(sys.argv[1])
    module_name, package_root = module_info_from_file(target)

    sys.path.insert(0, str(package_root))
    sys.argv = [str(target), *sys.argv[2:]]
    
    # if our __init__.py also imports the target module, we get an error
    # "RuntimeWarning: 'package.file' found in sys.modules after import of package 'package', but prior to execution of 'package.file'; this may result in unpredictable behaviour"
    # However, it does seem like it is working regardless, so let's silence the warnings and forget about them :)
    with warnings.catch_warnings(record=True) as warns:
        # alter_sys allows subprocesses (multiprocessing module) to import the state correctly without needing to update their module name manually
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        # if len(warns) and warns[0].category == RuntimeWarning:
        #     print("Encountered runtime warning")


if __name__ == "__main__":
    main()