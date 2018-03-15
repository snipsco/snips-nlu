.. _installation:

Installation
============

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


.. note::

   Currently we have built binaries (wheels) for ``snips-nlu`` and its
   dependencies for MacOS and Linux x86_64. If you use different
   architectures/os you will need to build these dependencies from sources
   which means you will need to install
   `setuptools_rust <https://github.com/PyO3/setuptools-rust>`_ and
   `Rust <https://www.rust-lang.org/en-US/install.html>`_.

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


.. _virtual environment: https://virtualenv.pypa.io/