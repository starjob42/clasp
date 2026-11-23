"""Collection guard for the published campaign directory.

``source_v1/`` is the immutable campaign source package: a byte-for-byte copy of
the files listed in its ``SOURCE_MANIFEST.sha256``, used to reproduce the runs
that produced the committed evidence. It deliberately carries its own copies of
the campaign tests, so collecting this directory would import two modules with
the same basename. Ignore the frozen copy here; the live tests beside this file
are the ones to run.
"""

collect_ignore = ["source_v1"]
