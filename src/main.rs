// TODO: use a more generic numeric data type for the data points throughout the lib
// TODO: proper dynamic tiling
// TODO: support dynamic filtering by line-tag
// TODO: move away from a veq for data point storage and instead use a Deque
// TODO: use your own hsv -> rgb converter

#![feature(mpmc_channel)]
#![feature(let_chains)]

use cli_log::*;
use hsv::hsv_to_rgb;
use std::collections::HashMap;
use std::str::SplitWhitespace;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use std::{io, thread};

use color_eyre::{Result, Section};
use ratatui::{
    crossterm::event::{self, Event, KeyCode},
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols::{self, Marker},
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, LegendPosition},
    DefaultTerminal, Frame,
};

use clap::Parser;

fn main() -> Result<()> {
    color_eyre::install()?;
    init_cli_log!();

    let args = Args::parse();
    let terminal = ratatui::init();
    let app_result = App::new(args).run(terminal);
    ratatui::restore();
    app_result
}

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// number of rows to display panels in
    #[arg(short, long)]
    rows: u32,
    /// number of cols to display panels in
    #[arg(short, long)]
    cols: u32,
    /// max points to keep in each graph
    #[arg(long)]
    max_points: Option<usize>,
}

struct App {
    stdin_rx: mpsc::Receiver<String>,
    args: Args,
    state: State,
}

/// Uses blocking sends to forward lines received over stdin
fn stdin_channel() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::sync_channel(10000);
    thread::spawn(move || loop {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        tx.send(buffer).unwrap();
    });

    rx
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            stdin_rx: stdin_channel(),
            args,
            state: HashMap::new(),
        }
    }

    fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(10);
        let mut last_tick = Instant::now();

        loop {
            terminal.draw(|frame| self.draw(frame))?;

            self.process_stdin_data();

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        return Ok(());
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }
    }

    fn process_stdin_data(&mut self) {
        // we only care about the successful case
        // we want to keep the graphs up even if the input channel died
        // TODO: implement a popup + some status line to inform the user that no new data will arrive on disconnect
        while let Ok(msg) = self.stdin_rx.try_recv() {
            parse_log_line(&msg).and_then(|intermediate_representation| {
                update_state(&mut self.state, &intermediate_representation).ok()
            });
        }

        for (_, lines) in self.state.iter_mut() {
            for (_, data) in lines.iter_mut() {
                if self
                    .args
                    .max_points
                    .is_some_and(|max_points| data.len() > max_points)
                {
                    data.drain(..(data.len() - self.args.max_points.unwrap()));
                }
            }
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let panels = {
            let rows = Layout::vertical(vec![
                Constraint::Ratio(1, self.args.rows);
                self.args.rows as usize
            ])
            .split(frame.area());

            let slices = (*rows).iter().map(|row| {
                Layout::horizontal(vec![
                    Constraint::Ratio(1, self.args.cols);
                    self.args.cols as usize
                ])
                .split(*row)
            });

            let mut out = Vec::new();
            for rc_slice in slices {
                out.extend_from_slice(&rc_slice);
            }

            out
        };

        let mut it = panels.iter();

        for (graph_title, lines) in self.state.iter() {
            if let Some(area) = it.next() {
                render_line_chart(frame, *area, graph_title, lines);
            }
        }
    }
}

struct MinMax {
    min: f64,
    max: f64,
}

impl MinMax {
    fn range(&self) -> f64 {
        self.max - self.min
    }

    fn bounds(&self, buffer: f64) -> (f64, f64) {
        assert!(buffer >= 0.0);

        let padding = self.range() * buffer;

        (self.min - padding, self.max + padding)
    }

    fn gen_labels(&self, n_labels: usize, buffer: f64) -> Vec<String> {
        assert!(n_labels > 2);

        let (min_bound, max_bound) = self.bounds(buffer);

        let range = max_bound - min_bound;

        let distance = range / (n_labels - 1) as f64;

        (0..n_labels)
            .map(|i| (min_bound + (i as f64 * distance)).to_string())
            .collect()
    }

    fn combine(&mut self, other: &MinMax) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

// TODO do this in one pass
fn get_min_max<'a>(it: impl Iterator<Item = &'a f64> + Clone) -> Option<MinMax> {
    let min = it.clone().min_by(|a, b| a.partial_cmp(b).unwrap());
    let max = it.clone().max_by(|a, b| a.partial_cmp(b).unwrap());

    if let (Some(min), Some(max)) = (min, max) {
        return Some(MinMax {
            min: *min,
            max: *max,
        });
    }

    None
}

struct Bounds {
    x: MinMax,
    y: MinMax,
}

impl Bounds {
    fn combine(&mut self, other: &Bounds) {
        self.x.combine(&other.x);
        self.y.combine(&other.y);
    }
}

impl From<&Vec<DataPoint>> for Bounds {
    fn from(data: &Vec<DataPoint>) -> Self {
        assert!(!data.is_empty());

        let min_max_x = get_min_max(data.iter().map(|(x, _)| x));
        let min_max_y = get_min_max(data.iter().map(|(_, y)| y));

        Bounds {
            x: min_max_x.unwrap(),
            y: min_max_y.unwrap(),
        }
    }
}

fn mk_dataset<'a>(
    line_name: &'a str,
    data: &'a [DataPoint],
    (r, g, b): (u8, u8, u8),
) -> Dataset<'a> {
    Dataset::default()
        .name(line_name)
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Rgb(r, g, b)))
        .graph_type(GraphType::Line)
        .data(data)
}

fn render_line_chart(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    graph: &HashMap<String, Vec<DataPoint>>,
) {
    let datasets = {
        let mut datasets = vec![];
        for (i, (line_name, line_data)) in graph.iter().enumerate() {
            let colour = hsv_to_rgb((360.0 * i as f64) / graph.len() as f64, 1.0, 1.0);
            let dataset = mk_dataset(line_name, line_data, colour);

            datasets.push(dataset);
        }

        datasets
    };

    let bounds = {
        let mut it = graph.iter().map(|(_, data)| Bounds::from(data));

        let mut acc = it.next().unwrap();

        for bound in it {
            acc.combine(&bound);
        }

        acc
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(Line::from(title).cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().gray())
                // TODO: use the widest bounds here so all the graphs have the same time domain
                .bounds(bounds.x.bounds(0.0).into())
                .labels(bounds.x.gen_labels(3, 0.0)),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().gray())
                .bounds(bounds.y.bounds(0.0).into())
                .labels(bounds.y.gen_labels(3, 0.0)),
        )
        .legend_position(Some(LegendPosition::TopLeft))
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

    frame.render_widget(chart, area);
}

// TODO: make this use a &str instead of a string
#[derive(Clone, Debug, PartialEq, Eq)]
struct IntermediateLogRepresentation {
    x: String,
    line_tags: Vec<String>,
    graphs: Vec<String>,
    lines: Vec<(String, String)>,
}

fn parse_log_line(log_line: &str) -> Option<IntermediateLogRepresentation> {
    #[cfg(test)]
    println!("got: {log_line:?}");
    let key_values = log_line.trim().split(',');

    let mut x_val = None;

    let mut line_tags = vec![];
    let mut graphs = vec![];
    let mut lines = vec![];

    for key_val in key_values {
        let key_val = key_val.trim();

        let splits: Vec<&str> = key_val.split('=').collect();

        if splits.len() != 2 {
            #[cfg(test)]
            println!("Skipping {key_val:?} as not an '=' seperated key value pair");
            return None;
        }

        let (key, val) = (splits[0], splits[1]);

        match key.rsplit_once('.') {
            Some((_, "x")) => {
                if let Some(_) = x_val {
                    #[cfg(test)]
                    println!("Skipping log line due to duplicate '.x' values '{log_line:?}'");
                    return None;
                } else {
                    x_val = Some(val.to_string());
                }
            }
            Some((_, "linetag")) => line_tags.push(val.to_owned()),
            Some((_, "graph")) => graphs.push(val.to_owned()),
            Some((prefix, "line")) => {
                lines.push((prefix.to_string(), val.to_owned()));
            }
            _ => {
                #[cfg(test)]
                println!("Skippig {key_val:?} as the key does not end with a known suffix");
            }
        }
    }

    if x_val.is_none() {
        trace!("Skipping log line due to no '.x' value {log_line}");
        return None;
    }

    Some(IntermediateLogRepresentation {
        x: x_val.unwrap(),
        line_tags,
        graphs,
        lines,
    })
}

// updates the global registry of data points
type GraphName = String;
type LineName = String;
type DataPoint = (f64, f64);

type State = HashMap<GraphName, HashMap<LineName, Vec<DataPoint>>>;

fn update_state(state: &mut State, new_datum: &IntermediateLogRepresentation) -> Result<()> {
    let x = new_datum.x.parse::<f64>()?;

    let line_prefix = new_datum.line_tags.join("-");

    for graph in new_datum.graphs.iter() {
        // TODO: skip this extra lookup
        if !state.contains_key(graph) {
            state.insert(graph.clone(), HashMap::new());
        }

        let graph = state.get_mut(graph).unwrap();

        for (line, y) in new_datum.lines.iter() {
            let line_name = line_prefix.clone() + "-" + &line;

            let y = {
                let y = y.parse::<f64>();
                if y.is_err() {
                    continue;
                }

                y.unwrap()
            };

            // TODO: skip extra lookup
            if !graph.contains_key(&line_name) {
                graph.insert(line_name.clone(), vec![]);
            }

            let data = graph.get_mut(&line_name).unwrap();

            data.push((x, y));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    const EXAMPLE_LOGS: &'static str = "
ts=2025-06-08T01:12:25.595348617+08:00, ns.x=1749316345595348617, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=34272, p50-ns.line=34273, p90-ns.line=34274, p99-ns.line=34275, pMax-ns.line=34276
ts=2025-06-08T01:12:25.595395830+08:00, ns.x=1749316345595395830, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=39822, p50-ns.line=39822, p90-ns.line=39822, p99-ns.line=39822, pMax-ns.line=0
ts=2025-06-08T01:12:25.595395830+08:00, ns.x=1749316345595395830, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=7391, p50-ns.line=7391, p90-ns.line=7391, p99-ns.line=7391, pMax-ns.line=0
ts=2025-06-08T01:12:25.595408051+08:00, ns.x=1749316345595408051, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=6120, p50-ns.line=6120, p90-ns.line=6120, p99-ns.line=6120, pMax-ns.line=0
ts=2025-06-08T01:12:25.595408051+08:00, ns.x=1749316345595408051, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=6101, p50-ns.line=6101, p90-ns.line=6101, p99-ns.line=6101, pMax-ns.line=0
ts=2025-06-08T01:12:25.595417751+08:00, ns.x=1749316345595417751, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=4080, p50-ns.line=4080, p90-ns.line=4080, p99-ns.line=4080, pMax-ns.line=0
ts=2025-06-08T01:12:25.595417751+08:00, ns.x=1749316345595417751, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=5620, p50-ns.line=5620, p90-ns.line=5620, p99-ns.line=5620, pMax-ns.line=0
ts=2025-06-08T01:12:25.595428712+08:00, ns.x=1749316345595428712, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=4730, p50-ns.line=4730, p90-ns.line=4730, p99-ns.line=4730, pMax-ns.line=0
ts=2025-06-08T01:12:25.595428712+08:00, ns.x=1749316345595428712, type.graph=EXECUTION_RUN, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=6231, p50-ns.line=6231, p90-ns.line=6231, p99-ns.line=6231, pMax-ns.line=0
ts=2025-06-08T01:12:25.595436992+08:00, ns.x=1749316345595436992, type.graph=EXECUTION_DOWNTIME, exchange.linetag=binf, symbol.linetag=PSWP-SOL/USDT, p0-ns.line=2720, p50-ns.line=2720, p90-ns.line=2720, p99-ns.line=2720, pMax-ns.line=0
";

    fn state_after_log_line_parse() -> State {
        let mut state = HashMap::new();

        let mut execution_downtime = HashMap::new();

        execution_downtime.insert(
            "binf-PSWP-SOL/USDT-p0-ns".to_owned(),
            vec![
                (1749316345595395830.0, 39822.0),
                (1749316345595408051.0, 6120.0),
                (1749316345595417751.0, 4080.0),
                (1749316345595428712.0, 4730.0),
                (1749316345595436992.0, 2720.0),
            ],
        );
        execution_downtime.insert(
            "binf-PSWP-SOL/USDT-p50-ns".to_owned(),
            vec![
                (1749316345595395830.0, 39822.0),
                (1749316345595408051.0, 6120.0),
                (1749316345595417751.0, 4080.0),
                (1749316345595428712.0, 4730.0),
                (1749316345595436992.0, 2720.0),
            ],
        );
        execution_downtime.insert(
            "binf-PSWP-SOL/USDT-p90-ns".to_owned(),
            vec![
                (1749316345595395830.0, 39822.0),
                (1749316345595408051.0, 6120.0),
                (1749316345595417751.0, 4080.0),
                (1749316345595428712.0, 4730.0),
                (1749316345595436992.0, 2720.0),
            ],
        );
        execution_downtime.insert(
            "binf-PSWP-SOL/USDT-p99-ns".to_owned(),
            vec![
                (1749316345595395830.0, 39822.0),
                (1749316345595408051.0, 6120.0),
                (1749316345595417751.0, 4080.0),
                (1749316345595428712.0, 4730.0),
                (1749316345595436992.0, 2720.0),
            ],
        );
        execution_downtime.insert(
            "binf-PSWP-SOL/USDT-pMax-ns".to_owned(),
            vec![
                (1749316345595395830.0, 0.0),
                (1749316345595408051.0, 0.0),
                (1749316345595417751.0, 0.0),
                (1749316345595428712.0, 0.0),
                (1749316345595436992.0, 0.0),
            ],
        );

        state.insert("EXECUTION_DOWNTIME".to_owned(), execution_downtime);

        let mut execution_run = HashMap::new();

        execution_run.insert(
            "binf-PSWP-SOL/USDT-p0-ns".to_owned(),
            vec![
                (1749316345595348617.0, 34272.0),
                (1749316345595395830.0, 7391.0),
                (1749316345595408051.0, 6101.0),
                (1749316345595417751.0, 5620.0),
                (1749316345595428712.0, 6231.0),
            ],
        );

        execution_run.insert(
            "binf-PSWP-SOL/USDT-p50-ns".to_owned(),
            vec![
                (1749316345595348617.0, 34273.0),
                (1749316345595395830.0, 7391.0),
                (1749316345595408051.0, 6101.0),
                (1749316345595417751.0, 5620.0),
                (1749316345595428712.0, 6231.0),
            ],
        );

        execution_run.insert(
            "binf-PSWP-SOL/USDT-p90-ns".to_owned(),
            vec![
                (1749316345595348617.0, 34274.0),
                (1749316345595395830.0, 7391.0),
                (1749316345595408051.0, 6101.0),
                (1749316345595417751.0, 5620.0),
                (1749316345595428712.0, 6231.0),
            ],
        );

        execution_run.insert(
            "binf-PSWP-SOL/USDT-p99-ns".to_owned(),
            vec![
                (1749316345595348617.0, 34275.0),
                (1749316345595395830.0, 7391.0),
                (1749316345595408051.0, 6101.0),
                (1749316345595417751.0, 5620.0),
                (1749316345595428712.0, 6231.0),
            ],
        );

        execution_run.insert(
            "binf-PSWP-SOL/USDT-pMax-ns".to_owned(),
            vec![
                (1749316345595348617.0, 34276.0),
                (1749316345595395830.0, 0.0),
                (1749316345595408051.0, 0.0),
                (1749316345595417751.0, 0.0),
                (1749316345595428712.0, 0.0),
            ],
        );

        state.insert("EXECUTION_RUN".to_owned(), execution_run);

        state
    }

    /// parse all of the logs into the final graph structure
    #[test]
    fn test_parse_example_logs() {
        let mut state = HashMap::new();

        for line in EXAMPLE_LOGS.lines() {
            let Some(intermediate_representation) = parse_log_line(line) else {
                continue;
            };

            let _ = update_state(&mut state, &intermediate_representation);
        }

        let result = state_after_log_line_parse();
        assert_eq!(
            state["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p0-ns"],
            result["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p0-ns"]
        );
        assert_eq!(
            state["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p50-ns"],
            result["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p50-ns"]
        );
        assert_eq!(
            state["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p90-ns"],
            result["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p90-ns"]
        );
        assert_eq!(
            state["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p99-ns"],
            result["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-p99-ns"]
        );
        assert_eq!(
            state["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-pMax-ns"],
            result["EXECUTION_RUN"]["binf-PSWP-SOL/USDT-pMax-ns"]
        );

        assert_eq!(
            state["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p0-ns"],
            result["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p0-ns"]
        );
        assert_eq!(
            state["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p50-ns"],
            result["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p50-ns"]
        );
        assert_eq!(
            state["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p90-ns"],
            result["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p90-ns"]
        );
        assert_eq!(
            state["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p99-ns"],
            result["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-p99-ns"]
        );
        assert_eq!(
            state["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-pMax-ns"],
            result["EXECUTION_DOWNTIME"]["binf-PSWP-SOL/USDT-pMax-ns"]
        );

        assert_eq!(state["EXECUTION_RUN"], result["EXECUTION_RUN"]);
        assert_eq!(state, result);
    }

    #[test]
    fn test_intermediate_parse() {
        for (line, result) in EXAMPLE_LOGS.lines().skip(1).take(1).zip(
            vec![IntermediateLogRepresentation {
                x: "1749316345595348617".to_owned(),
                line_tags: vec!["binf".to_string(), "PSWP-SOL/USDT".to_string()],
                graphs: vec!["EXECUTION_RUN".to_string()],
                lines: vec![
                    ("p0-ns".to_string(), "34272".to_owned()),
                    ("p50-ns".to_string(), "34273".to_owned()),
                    ("p90-ns".to_string(), "34274".to_owned()),
                    ("p99-ns".to_string(), "34275".to_owned()),
                    ("pMax-ns".to_string(), "34276".to_owned()),
                ],
            }]
            .iter(),
        ) {
            assert_eq!(parse_log_line(line), Some(result.clone()));
        }
    }
}
