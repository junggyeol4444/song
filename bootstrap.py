#!/usr/bin/env python3
"""
처음 켤 때 필요한 것을 확인하고 설치한다.

윈도우 배치 파일에서 한글을 출력하면 깨진다. 한국 윈도우의 기본 코드페이지가
CP949 인데 파일은 UTF-8 로 저장되기 때문이다. chcp 로 바꿔도 환경에 따라
동작이 다르다. 그래서 배치 파일은 영문 명령만 담고, 사람이 읽을 안내는
전부 여기서 출력한다. 파이썬은 인코딩을 제대로 다룬다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = (
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("soundfile", "soundfile", True),
    ("mido", "mido", True),
    ("PySide6", "PySide6-Essentials", False),
    ("sounddevice", "sounddevice", False),
)


def setup_console() -> None:
    """윈도우 콘솔에서 한글이 깨지지 않게 한다."""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass


def missing_packages() -> tuple[list[str], list[str]]:
    """(꼭 필요한데 없는 것, 있으면 좋은데 없는 것)"""
    required: list[str] = []
    optional: list[str] = []
    for module, package, is_required in REQUIRED:
        try:
            __import__(module)
        except Exception:
            (required if is_required else optional).append(package)
    return required, optional


def install(packages: list[str]) -> bool:
    print(f"\n설치합니다: {', '.join(packages)}")
    print("처음 한 번만 시간이 걸립니다. 인터넷 연결이 필요합니다.\n")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        cwd=ROOT,
    )
    return result.returncode == 0


def main() -> int:
    setup_console()
    print("MYVOCAL Studio")
    print(f"파이썬 {sys.version.split()[0]}\n")

    if sys.version_info < (3, 10):
        print("파이썬 3.10 이상이 필요합니다.")
        print("  https://www.python.org/downloads/ 에서 새로 받아 주세요.")
        print("  설치할 때 'Add Python to PATH' 를 반드시 체크하세요.")
        return 1

    required, optional = missing_packages()

    if required:
        print("꼭 필요한 것이 없습니다:")
        for name in required:
            print(f"  · {name}")
        if not install(required):
            print("\n설치에 실패했습니다.")
            print("인터넷 연결을 확인하거나, 아래를 직접 실행해 보세요:")
            print(f"  {sys.executable} -m pip install {' '.join(required)}")
            return 1
        required, optional = missing_packages()
        if required:
            print(f"\n설치했는데도 불러올 수 없습니다: {', '.join(required)}")
            return 1

    if optional:
        print("없어도 되지만 있으면 좋은 것:")
        for name in optional:
            reason = {
                "PySide6-Essentials": "화면 (없으면 명령줄로만 쓸 수 있습니다)",
                "sounddevice": "소리 재생 (없으면 파일로 내보내서 들어야 합니다)",
            }.get(name, "")
            print(f"  · {name} — {reason}")
        answer = input("\n설치할까요? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes", "ㅛ"):
            install(optional)

    print("\n준비됐습니다.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
