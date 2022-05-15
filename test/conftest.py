import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def datadir(request):
    filename = request.module.__file__
    data_dir = Path(filename).parent / "data"

    with tempfile.TemporaryDirectory() as tempdir:
        tmp_data_dir = Path(tempdir) / "data"
        if data_dir.exists():
            shutil.copytree(data_dir, tmp_data_dir)
        else:
            tmp_data_dir.mkdir()

        yield tmp_data_dir

    return
