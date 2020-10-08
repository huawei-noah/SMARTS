.. _adapters:

Adapters
========

TODO: needs to be fleshed out

.. graphviz::

  digraph foo {
      rankdir = "LR"
      "Policy" -> "SMARTS" [ label="policy action" ];
      "SMARTS" -> "Policy" [ label="observation" ];
      "SMARTS" -> "Policy" [ label="reward" ];
      "SMARTS" -> "Policy" [ label="info" ];
   }
