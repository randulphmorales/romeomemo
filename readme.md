The package "romeomemo" contains Python code used
for the computation of emissions for methane point
sources used for Morales et. al (2021) AMT publication
(amt-2021-314).

Installing
==========
To install the requirements use:
```
$ python -m pip install -r requirements.txt
```

To install `romeomeo` you can also use pip. If you want your
working copy known to your python session, you
can install a local (editable) development version adding
`-e` to the following command:

```
$ pip -m install /path/to/romeomemo
```
which will "compile" the code locally but creates a link in
your local library folder. The link can be removed with:
```
$ pip -m uninstall romeomemo
```

Virtual Environment
===================
To create a virtual environment using anaconda use:

```
$ conda env create -f romeomemo.yml
```

Running the code
================
**Cluster-based kriging**
The main script to calculate emissions using the cluster-kriging
approach is `main_clusterkrige.py`. It needs a configuration file `config.py` which contains the paths of the input files needed to compute emissions fluxes. To run the code:
```
python main_clusterkrige.py config.py
```

**Ordinary kriging**
To compute emission fluxes using ordinary kriging:
```
python main_ordinarykrige.py config.py
```
