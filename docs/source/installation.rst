.. _installation:

Installation
============

System requirements
-------------------
- 64-bit Linux, MacOS >= 10.11, 64-bit Windows
- Python 2.7 or Python >= 3.4
- RAM: Snips NLU will typically use between 100MB and 200MB of RAM, depending on the language and the size of the dataset.


Python Version
--------------

We recommend using the latest version of Python 3. Snips NLU supports Python
3.4 and newer as well as Python 2.7.


Install Snips NLU
-----------------

It is recommended to use a `virtual environment`_ and activate it before
installing Snips NLU in order to manage your project dependencies properly.

Snips NLU can be installed via pip with the following command:

.. code-block:: sh

    pip install snips-nlu


We currently have pre-built binaries (wheels) for ``snips-nlu`` and its
dependencies for MacOS (10.11 and later), Linux x86_64 and Windows 64-bit. If
you use a different architecture/os you will need to build these dependencies
from sources which means you will need to install
`setuptools_rust <https://github.com/PyO3/setuptools-rust>`_ and
`Rust <https://www.rust-lang.org/en-US/install.html>`_ before running the
``pip install snips-nlu`` command.

Language resources
------------------

Snips NLU relies on `language resources`_ which must be downloaded beforehand.
To fetch the resources for a specific language, run the following command:

.. code-block:: sh

    python -m snips_nlu download <language>

Or simply:

.. code-block:: sh

    snips-nlu download <language>

The list of supported languages is described :ref:`here <languages>`.


Extra dependencies
------------------

-------
Metrics
-------

If at some point you want to compute metrics, you will need some extra
dependencies that can be installed via:

.. code-block:: sh

    pip install 'snips-nlu[metrics]'

-----
Tests
-----

.. code-block:: sh

    pip install 'snips-nlu[test]'

-------------
Documentation
-------------

.. code-block:: sh

    pip install 'snips-nlu[doc]'


.. _virtual environment: https://virtualenv.pypa.io
.. _language resources: https://github.com/snipsco/snips-nlu-language-resources