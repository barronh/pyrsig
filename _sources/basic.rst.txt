.. pyrsig documentation master file, created by
   sphinx-quickstart on Fri Mar 17 21:46:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyrsig User's Guide
===================

Python interface to RSIG Web API.

The key value of `pyrsig` is to present RSIG data in pandas DataFrames and
xarray Datasets. This makes it easy to do advanced analyses in a pythonic
way. Example analyses are provided, but the sky is the limit.

RSIG connects you to air quality data. The figure below highlights how RSIG
operates as a central access point for air quality data from many data
partners. `pyrsig` is one app that you can choose to get data from RSIG and
all its partners.

.. image:: _static/rsig_network.png
    :alt: RSIG Interargency Networked Data

variables and petabytes of data.
More information about RSIG and RSIG datasets is available at EPA's `RSIG website`_.

.. _RSIG website: https://www.epa.gov/hesc/remote-sensing-information-gateway


Getting Started
---------------

The best way to get started is to install (see below) and then explore the
examples gallery.


Installation
------------

`pyrsig` is avalable through pypi.org, but is still in rapid development. You
can get the latest release from pypi via the command below.

.. code-block::

    pip install pyrsig

Or, you can get the latest version with this command.

.. code-block::

    pip install git+https://github.com/barronh/pyrsig.git


Issues
------

If you're having any problems, open an issue on github.

https://github.com/barronh/pyrsig/issues


Quick Links
-----------

* :doc:`auto_examples/index`
* :doc:`api/pyrsig`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents

   self
   auto_examples/index
   api/pyrsig
