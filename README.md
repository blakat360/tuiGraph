# What is this?

The idea is that the app stores a hashmap of graph -> line -> data.

These are created using the suffixes:
- linetag: prepended to any "line" entries
- line
- graph

So something like this:
```
ts=2025-06-08T01:12:25.595348617+08:00, ns.x=1749316345595348617, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=34272, p50-ns.line=34273, p90-ns.line=34274, p99-ns.line=34275, pMax-ns.line=34276
```

defines 1 graph with 5 lines (one for each percentile).

The app keeps showing you that one graph until it sees another log line:

```
ts=2025-06-08T01:12:25.595395830+08:00, ns.x=1749316345595395830, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=39822, p50-ns.line=39822, p90-ns.line=39822, p99-ns.line=39822, pMax-ns.line=0
```

Which defines a second "EXECUTION_RUN" graph with another 5 lines.
Changing the line tag value to lets say "bina" would create a new line on the same graph.
The main pitch is no persister process or SQL queries needed for a grafana-like experience, and if you want to change what you report in your app, all your telemetry graphs pull in the new data alongside the old without any changes.

# To run with dev example with app logging

`cat example_logs.txt | sed 's/.*STATS://g' | TUI_GRAPH_LOG=debug cargo run -- -r 2 -c 1`
