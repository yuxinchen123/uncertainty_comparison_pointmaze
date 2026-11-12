# probes

One record per doubling series, `<workload>__<card>__<YYYY-MM-DD-HH-MM>.json` (time in Pacific), written by
`../series.py`: the workload, the card and its device name, the host, and one point per copy count (seconds per
iteration, the five throughput quantities, peak card memory, or the reason the size has no measurement).
`logs/<series>/` holds the output of each size.
