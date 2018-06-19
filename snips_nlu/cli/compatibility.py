import sys

is_windows = sys.platform.startswith('win')

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3


def create_symlink(orig, dest):
    if PY2 and is_windows:
        import subprocess
        subprocess.call(['mklink', '/d', str(orig).decode("utf8"),
                         str(dest).decode("utf8")], shell=True)
    else:
        orig.symlink_to(dest)
