Choosing k
==========

``k`` defines the current extraction. It must satisfy ``2 <= k <= k_max``.

.. code-block:: python

   clusterer.fit(X, k=10)

Interpretation
--------------

Larger ``k`` values generally smooth the density estimate more strongly.
Smaller ``k`` values make the hierarchy more sensitive to local structure.

Core-SG makes it cheap to compare several valid ``k`` values after the initial
build, so a practical workflow is to fit once at the largest candidate and
inspect labels, probabilities, persistence, and tree artifacts across the
candidate list.
