===============================
mri2mri
===============================

.. image:: https://img.shields.io/travis/uw-biomedical-ml/mri2mri.svg
        :target: https://travis-ci.org/uw-biomedical-ml/mri2mri

.. image:: https://img.shields.io/pypi/v/mri2mri.svg
        :target: https://pypi.python.org/pypi/mri2mri


convert one type of MRI to another

* Free software: 3-clause BSD license
* Documentation: https://uw-biomedical-ml.github.io/mri2mri.

Installation
------------

To install the software, you will first need to install `pytorch <https://pytorch.org/>`_

Please follow the installation instructions on their website first.

Then, download the source, navigate your shell to the top-level source
directory and run::

        pip install -r requirements.txt
        pip install .

If you intend to make changes and work on development of MRI2MRI, please
run::

        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .


Usage
--------

We only support T1 weighted to T2 weighted, and T2 weighted to T1 weighted translation right now. For example::

        mri2mri --input path_to_input_T1.nii.gz --which_transform t1w2t2w

You can use ``mri2mri --help`` to see more options
