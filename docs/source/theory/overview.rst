Overview
========

Core-SG separates reusable graph construction from repeated hierarchy
extraction.

.. raw:: html

   <div class="docs-diagram">
     <div class="docs-diagram-title">Estimator-level flow</div>
     <div class="diagram-flow">
       <div class="diagram-step accent">input data X</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step secondary">CoreSGClusterer.fit(X, k)</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step warning">build or reuse core_sg_</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step">extract hierarchy for k</div>
       <div class="diagram-arrow">→</div>
       <div class="diagram-step">labels_, probabilities_, trees</div>
     </div>
   </div>

The high-level flow is:

1. ``CoreSGClusterer`` receives ``X`` and the target ``k``;
2. on the first call, it builds reusable support at ``k_max``;
3. on later calls, it reuses the existing ``core_sg_`` object;
4. for each target ``k``, it extracts the current hierarchy;
5. it exposes labels, probabilities, persistence values, and tree objects on
   the estimator.

The main benefit appears when several ``k`` values are needed for the same
dataset.

The default exact ``algorithm="core-sg"`` path demonstrates the reuse model but
still depends on dense pairwise distance information, which creates a practical
``n_samples`` limit. The approximate ``algorithm="score-sg"`` path is designed
to address that scaling limitation by replacing dense all-pairs construction
with a sparse approximate-neighbor support graph.
