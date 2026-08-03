# Test report v0.7.3

## Automated test suite

```text
51 passed
```

Coverage added for:

- hidden-oracle Δon/ΔS and success calculation
- negative-episode abstention
- unsafe acceptance penalty
- Stage C NEEDS_VALIDATION behavior
- aggregate summary
- greedy and reproducible random-valid baselines

## Additional smoke tests

- Built a synthetic pair cache
- Generated a held-out positive evaluation snapshot
- Rebuilt visible MMP rules after endpoint removal
- Ran the seed-only baseline over five repeated seeds
- Produced episode metrics, policy summary, and run manifest

The synthetic smoke data are not included as scientific evaluation results.
