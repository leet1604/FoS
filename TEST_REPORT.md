# Test report

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
..........                                                               [100%]
10 passed
```

The live ChEMBL path was not executed in the offline artifact environment. Live provider syntax/imports were compiled; network behavior should be validated on the user's server with the existing ChEMBL cache.
