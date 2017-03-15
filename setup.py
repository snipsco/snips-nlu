import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.install import install

from snips_nlu.utils import ROOT_PATH


class SnipsNLUInstall(install):
    def do_egg_install(self):
        print("Installing built-ins...")
        env = os.environ.copy()
        env["PATH"] = "~/.cargo/bin:" + env["PATH"]

        queries_cli_path = os.path.join(ROOT_PATH, "snips-queries-rust")
        if os.path.exists(queries_cli_path):
            shutil.rmtree(queries_cli_path)

        subprocess.call(["cd", ROOT_PATH])

        git_project_url = "git@github.com:snipsco/snips-queries-rust.git"
        subprocess.call(["git", "clone", "--recursive", git_project_url])
        print("Building built-ins...")

        subprocess.call(["cargo install protobuf"], env=env, shell=True)
        try:
            args = (["cargo build"])
            cwd = os.path.join(ROOT_PATH, "snips-queries-rust", "queries-cli")
            subprocess.check_call(args, env=env, cwd=cwd, shell=True)
        except subprocess.CalledProcessError as e:
            msg = "cargo failed with code: %d\n%s" % (e.returncode, e.output)
            raise Exception(msg)
        except OSError:
            raise Exception("Unable to execute 'cargo' - this package "
                            "requires rust to be installed and cargo to be on "
                            "the PATH")

        install.do_egg_install(self)


setup(name="snips_nlu",
      version="0.0.1",
      description="",
      author="Clement Doumouro",
      author_email="clement.doumouro@snips.ai",
      url="",
      download_url="",
      license="MIT",
      install_requires=["enum34", "sklearn-crfsuite"],
      packages=["snips_nlu",
                "snips_nlu.entity_extractor",
                "snips_nlu.nlu_engine"],
      cmdclass={"install": SnipsNLUInstall},
      entry_points={},
      include_package_data=False,
      zip_safe=False)
