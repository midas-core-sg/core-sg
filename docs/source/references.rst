References
==========

This page lists the bibliographic and technical references used to document
Core-SG, its HDBSCAN-style hierarchy behavior, Score-SG, approximate nearest
neighbor construction, and noise-label post-processing.

Academic References
-------------------

Core-SG
~~~~~~~

Antonio Cavalcante Araujo Neto, Murilo Coelho Naldi, Ricardo J. G. B.
Campello, and Jörg Sander. *CORE-SG: Efficient Computation of Multiple MSTs
for Density-Based Methods*. In: 2022 IEEE 38th International Conference on
Data Engineering (ICDE), IEEE, pp. 951--964, 2022.
DOI: `10.1109/ICDE53745.2022.00076 <https://doi.org/10.1109/ICDE53745.2022.00076>`__.

HDBSCAN and Hierarchical Density Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leland McInnes and John Healy. *Accelerated Hierarchical Density Based
Clustering*. In: 2017 IEEE International Conference on Data Mining Workshops
(ICDMW), IEEE, pp. 33--42, 2017.
DOI: `10.1109/ICDMW.2017.12 <https://doi.org/10.1109/ICDMW.2017.12>`__.

Ricardo J. G. B. Campello, Davoud Moulavi, and Jörg Sander. *Density-Based
Clustering Based on Hierarchical Density Estimates*. In: Advances in Knowledge
Discovery and Data Mining, Springer, pp. 160--172, 2013.
DOI: `10.1007/978-3-642-37456-2_14 <https://doi.org/10.1007/978-3-642-37456-2_14>`__.

Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, and Jörg Sander.
*Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier
Detection*. ACM Transactions on Knowledge Discovery from Data, 10(1), 2015.
DOI: `10.1145/2733381 <https://doi.org/10.1145/2733381>`__.

Noise Handling and Density-Based Label Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jadson Castro Gertrudes, Arthur Zimek, Jörg Sander, and Ricardo J. G. B.
Campello. *A Unified View of Density-Based Methods for Semi-Supervised
Clustering and Classification*. Data Mining and Knowledge Discovery, 33,
1894--1952, 2019.
DOI: `10.1007/s10618-019-00651-1 <https://doi.org/10.1007/s10618-019-00651-1>`__.

Approximate Nearest Neighbors and PyNNDescent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wei Dong, Moses Charikar, and Kai Li. *Efficient k-Nearest Neighbor Graph
Construction for Generic Similarity Measures*. In: Proceedings of the 20th
International Conference on World Wide Web (WWW), ACM, pp. 577--586, 2011.
DOI: `10.1145/1963405.1963487 <https://doi.org/10.1145/1963405.1963487>`__.

Software and Documentation References
-------------------------------------

HDBSCAN
~~~~~~~

The Core-SG hierarchy adapter is documented against the HDBSCAN ecosystem and
its private tree-processing internals.

* HDBSCAN documentation: https://hdbscan.readthedocs.io/en/latest/
* HDBSCAN repository: https://github.com/scikit-learn-contrib/hdbscan

PyNNDescent
~~~~~~~~~~~

Score-SG uses PyNNDescent for approximate nearest-neighbor graph construction.

* PyNNDescent documentation: https://pynndescent.readthedocs.io/en/stable/
* PyNNDescent package metadata: https://pypi.org/project/pynndescent/

Scikit-Learn
~~~~~~~~~~~~

``CoreSGClusterer`` follows the scikit-learn estimator style for ``fit``,
``fit_predict``, parameter introspection, and cloning behavior.

* scikit-learn documentation: https://scikit-learn.org/stable/

Documentation Tooling
~~~~~~~~~~~~~~~~~~~~~

The GitHub Pages site is built with Sphinx and the Read the Docs theme.

* Sphinx documentation: https://www.sphinx-doc.org/
* sphinx-rtd-theme documentation: https://sphinx-rtd-theme.readthedocs.io/

Reference Usage Map
-------------------

* The Core-SG paper motivates the reusable support graph and repeated
  multi-``k`` MST extraction workflow.
* The HDBSCAN and hierarchical density clustering references motivate the
  hierarchy, condensed tree, persistence, and HDBSCAN-style output language.
* The density-based semi-supervised clustering reference informs the current
  noise-label reassignment discussion.
* The nearest-neighbor descent reference and PyNNDescent documentation inform
  the Score-SG approximate kNN construction discussion.
* The scikit-learn documentation informs the estimator-oriented API
  presentation.
