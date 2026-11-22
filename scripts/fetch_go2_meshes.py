# ruff: noqa: E501  (SHA-256 manifest lines)
"""Download the Go2's visual meshes (MuJoCo Menagerie) for rendering.

The URDF in resources/robots/go2 refers to COLLADA meshes that are not in
the repository (and MuJoCo cannot load COLLADA), so without these files the
Go2 renders as its collision shapes. This script downloads MuJoCo
Menagerie's unitree_go2 model (go2.xml + OBJ meshes, ~28 MB) at a pinned
commit into resources/robots/go2/mjcf/ (git-ignored) and checks every file
against a SHA-256 manifest. The simulator only uses its meshes and colors
for display (see gym/envs/base/urdf_visuals.py); masses, joints and
collisions still come from the URDF, so training is unaffected.

    uv run --frozen scripts/fetch_go2_meshes.py

The meshes are Unitree's, distributed under the BSD-3-Clause license in
resources/robots/go2/mjcf/LICENSE.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

REPOSITORY = "google-deepmind/mujoco_menagerie"
# Latest commit touching unitree_go2 when this script was written.
REVISION = "6da8b971844845de6ab27315aacc3a3fea4ddb8e"
SOURCE = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/unitree_go2"
TARGET = Path(__file__).resolve().parents[1] / "resources/robots/go2/mjcf"

MANIFEST = {
    "go2.xml": "50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a",
    "LICENSE": "7e2dfda59b98e38d123f539b59ecb637ffd8908d19e09f7765884a309f010d40",
    "assets/base_0.obj": "8cdc53719002a5ee737d867904e534a6d6fa930b616bdcc1b3131e577d4e46c0",
    "assets/base_1.obj": "39d2f4724ff64dd5a0640c308fd98902a2946660a8d033ec8dbf368fd5af0dd6",
    "assets/base_2.obj": "29cae463f95d678e4ede863c36c9b19b0d356b8340412add10670d70bdcc3095",
    "assets/base_3.obj": "f51ba43bf300e68957a0c1aa83eaf3e4926f4ac7e64a0d6c33ae5d21fcbcf5f6",
    "assets/base_4.obj": "49ddec0129546e5dd2075d5beacf7f207baa2860aeff7626818e64d2efbf3821",
    "assets/calf_0.obj": "4a818cd563c5b7e2a271504442d05fd777d38cbaa028d4a10ad29c9f87bb355e",
    "assets/calf_1.obj": "51950c3a153ae3208592df86ea11c9d6756f8f645ebca5caf6910c61aeb03817",
    "assets/calf_mirror_0.obj": "6dc0406207a1c08a5953fd5962aca1a270fd06593d30d100205e05c1b32d4af2",
    "assets/calf_mirror_1.obj": "f84f5af90b914633165beb3a5493ee499169144c5797dcf04c6eb5842f74b31a",
    "assets/foot.obj": "df9e78a7c0110d02557439010f2a5f11f0ec4fd907b32704145dbd6aa7affc0b",
    "assets/hip_0.obj": "cc29932ab4d251ce3f72389ec4c1f757e44eee138b87e772b648c2bc76b506cd",
    "assets/hip_1.obj": "3fd5bd9b2ef6abe622c0739d3c67eaa1679b19b47132951972c4892a779ea7cf",
    "assets/thigh_0.obj": "bb3edfe6a7b04f4f4dff1ce7b05d426ddce941e276cc6372fb542055893c71ec",
    "assets/thigh_1.obj": "3a631822f9f7d2114b1b79dcdff5abf95103630abb7d91d1f3b8b18117132ed2",
    "assets/thigh_mirror_0.obj": "07abaedce761f1dd48647384ac14ba1959276bc78d5a2a62051e8f1c6093933b",
    "assets/thigh_mirror_1.obj": "db984a3e983eabbc9b38183427cca9ab0bb3bdb049f9eab030e06edda36579d6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(target: Path = TARGET, force: bool = False) -> Path:
    """Download any missing or mismatched file; return the model path."""
    for name, digest in MANIFEST.items():
        path = target / name
        if path.is_file() and sha256(path) == digest and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        with urllib.request.urlopen(f"{SOURCE}/{name}", timeout=60) as response:
            temporary.write_bytes(response.read())
        if sha256(temporary) != digest:
            temporary.unlink()
            raise RuntimeError(f"{name}: checksum mismatch; not installed")
        temporary.replace(path)
        print(f"fetched {name}")
    return target / "go2.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download all files")
    args = parser.parse_args()
    try:
        model = fetch(force=args.force)
    except (OSError, RuntimeError) as error:
        sys.exit(f"could not fetch Go2 meshes: {error}")
    print(f"Go2 visual model ready: {model.relative_to(TARGET.parents[3])}")


if __name__ == "__main__":
    main()
