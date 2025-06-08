# To run with logging + nc as input source

`nc -l 0.0.0.0 5555 | TUI_GRAPH_LOG=debug cargo +nightly run -- -r 4 -c 4`
