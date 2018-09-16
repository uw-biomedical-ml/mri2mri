=====
Usage
=====

MRI2MRI provides a command line interface that takes a nifti file with 
one MR contrast and produces a nifti file with another image contrast. 

For example, download the following `file <https://figshare.com/articles/High_resolution_t1_weighted_image/848608>`_ 
to your computer. This is a high-resolution T1-weighted image published by 
`Kirstie Whitaker <https://whitakerlab.github.io/>`_. 

Here's what this file looks like when viewed in `Mango <http://ric.uthscsa.edu/mango/mango.html>`_:

.. image:: _static/orig_t1.png
   :width: 200px

We call the CLI, providing the `--transform` input as `t1w2t2w` which means that we will be 
transforming a T1-weighted image into a T2-weighted image.


.. code-block:: bash

    mri2mri --input t1.nii.gz --transform t1w2t2w --output t2_synthetic.nii.gz

The result is as follows:

.. image:: _static/synthetic_t2.png
   :width: 200px
