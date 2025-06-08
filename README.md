# To run with logging + nc as input source

`nc -l 0.0.0.0 5555 | TUI_GRAPH_LOG=debug cargo +nightly run -- -r 4 -c 4`

# To run with predefined input

```
TUI_GRAPH_LOG=debug cargo +nightly run -- -r 4 -c 4 << END
1 1
2 2
3.5 4.5
END
```

Or just cat something in
