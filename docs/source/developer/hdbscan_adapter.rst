HDBSCAN Adapter
===============

The adapter must be the only module that directly imports HDBSCAN private APIs.
``CoreSG`` and ``CoreSGClusterer`` should not import HDBSCAN private modules
directly.

The intended dependency direction is:

.. code-block:: text

   CoreSG -> core_sg.hdbscan_adapter -> HDBSCAN internals

Responsibilities
----------------

The adapter handles reference MST construction, single linkage conversion,
tree-to-label conversion, tree wrapper creation, and compatibility filtering
for private ``_tree_to_labels(...)`` keyword arguments.

This boundary localizes version risk. If HDBSCAN changes a private signature,
``core_sg/hdbscan_adapter.py`` should be the first place to update.
