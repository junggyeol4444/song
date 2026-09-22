#!/usr/bin/env python3
"""
윈도우 실행 파일 만들기.

    python build_windows.py

MYVOCAL Studio.exe 하나로 묶는다. 파이썬이 안 깔린 컴퓨터에서도 돌아간다.

왜 폴더 방식(--onedir)을 쓰는가
    한 파일(--onefile)로 묶으면 실행할 때마다 임시 폴더에 전부 풀어 놓는다.
    이 프로그램은 numpy, scipy, Qt 가 들어가서 200MB 가 넘으므로, 켤 때마다
    10초 넘게 걸리고 백신이 매번 검사한다. 폴더 방식은 처음부터 풀려 있어
    곧바로 켜진다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "MYVOCAL Studio"
ENTRY = ROOT / "main.py"


def check_requirements() -> list[str]:
    """빌드에 필요한 것이 있는지 본다."""
    problems: list[str] = []
    if sys.platform != "win32":
        problems.append(
            f"지금은 {sys.platform} 입니다. 윈도우 실행 파일은 윈도우에서 만들어야 합니다.\n"
            f"    (리눅스나 맥에서 만든 것은 윈도우에서 안 돌아갑니다)"
        )
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        problems.append("PyInstaller 가 없습니다.\n    pip install pyinstaller")
    for module in ("numpy", "scipy", "soundfile", "mido", "PySide6"):
        try:
            __import__(module)
        except ImportError:
            problems.append(f"{module} 이 없습니다.\n    pip install {module}")
    if not ENTRY.exists():
        problems.append(f"{ENTRY} 를 찾을 수 없습니다.")
    return problems


def build_arguments() -> list[str]:
    """PyInstaller 에 넘길 설정."""
    arguments = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",                    # 콘솔 창을 띄우지 않는다
        "--name", APP_NAME,
        # 이 프로그램이 실제로 쓰는 것만 넣는다. 안 쓰는 Qt 모듈까지 들어가면
        # 용량이 두 배가 되고 켜는 시간도 길어진다.
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "IPython",
        "--exclude-module", "pytest",
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.QtQuick",
        "--exclude-module", "PySide6.Qt3DCore",
        "--exclude-module", "PySide6.QtCharts",
        "--exclude-module", "PySide6.QtDataVisualization",
        # soundfile 과 sounddevice 는 딸린 DLL 을 자동으로 못 찾는 경우가 있다
        "--collect-binaries", "soundfile",
        "--hidden-import", "soundfile",
        "--hidden-import", "sounddevice",
        "--hidden-import", "scipy.signal",
        "--hidden-import", "scipy.special",
    ]
    icon = ROOT / "assets" / "icon.ico"
    if icon.exists():
        arguments += ["--icon", str(icon)]
    arguments.append(str(ENTRY))
    return arguments


def main() -> int:
    problems = check_requirements()
    if problems:
        print("빌드할 수 없습니다:\n")
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    print(f"{APP_NAME} 실행 파일을 만듭니다...\n")
    result = subprocess.run(build_arguments(), cwd=ROOT)
    if result.returncode != 0:
        print("\n빌드가 실패했습니다. 위 메시지를 확인해 주세요.")
        return result.returncode

    output = ROOT / "dist" / APP_NAME
    executable = output / f"{APP_NAME}.exe"
    if not executable.exists():
        print(f"\n빌드는 끝났는데 {executable} 이 없습니다.")
        return 1

    total = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    print(f"\n완성: {executable}")
    print(f"폴더 전체 크기: {total / 1024 / 1024:.0f} MB")
    print(f"\n{output} 폴더를 통째로 옮기면 다른 컴퓨터에서도 돌아갑니다.")
    print(f"{APP_NAME}.exe 만 따로 옮기면 안 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
