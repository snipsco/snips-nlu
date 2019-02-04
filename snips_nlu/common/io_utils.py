import errno
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile, ZIP_DEFLATED


def mkdir_p(path):
    """Reproduces the 'mkdir -p shell' command

    See
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(str(path))
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and path.is_dir():
            pass
        else:
            raise


@contextmanager
def temp_dir():
    tmp_dir = mkdtemp()
    try:
        yield Path(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir)


def unzip_archive(archive_file, destination_dir):
    with ZipFile(archive_file, "r", ZIP_DEFLATED) as zipf:
        zipf.extractall(str(destination_dir))
