# Celerity Legacy Branch: Buffer Manager Runtime

This is a fork of Celerity right before the runtime switched to IDAG scheduling,
and serves as a baseline to benchmark the old (single-GPU) buffer manager runtime.
The branch does include IDAG generator etc, but they are not used outside of tests.

Dependency upgrades and some other minor patches have been backported from master.
