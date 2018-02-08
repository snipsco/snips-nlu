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

Extra dependencies
------------------

If at some point you want to compute metrics, you will need some extra
dependencies that can be installed via:

.. code-block:: sh

    pip install 'snips-nlu[metrics]'


.. _virtual environment: https://virtualenv.pypa.io/