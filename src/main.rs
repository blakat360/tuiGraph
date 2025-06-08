// TODO: use a more generic numeric data type for the data points throughout the lib
// TODO: proper dynamic tiling
// TODO: support dynamic filtering by line-tag
// TODO: move away from a veq for data point storage and instead use a Deque

#![feature(mpmc_channel)]
#![feature(let_chains)]

use cli_log::*;
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
    signal1: SinSignal,
    data1: Vec<(f64, f64)>,
    signal2: SinSignal,
    data2: Vec<(f64, f64)>,
    window: [f64; 2],
    line_data: Vec<(f64, f64)>,
    stdin_rx: mpsc::Receiver<String>,
    args: Args,
}

/// Uses blocking sends to forward lines received over stdin
fn stdin_channel() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::sync_channel(0);
    thread::spawn(move || loop {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        tx.send(buffer).unwrap();
    });

    rx
}

#[derive(Clone)]
struct SinSignal {
    x: f64,
    interval: f64,
    period: f64,
    scale: f64,
}

impl SinSignal {
    const fn new(interval: f64, period: f64, scale: f64) -> Self {
        Self {
            x: 0.0,
            interval,
            period,
            scale,
        }
    }
}

impl Iterator for SinSignal {
    type Item = (f64, f64);
    fn next(&mut self) -> Option<Self::Item> {
        let point = (self.x, (self.x * 1.0 / self.period).sin() * self.scale);
        self.x += self.interval;
        Some(point)
    }
}

impl App {
    fn new(args: Args) -> Self {
        let mut signal1 = SinSignal::new(0.2, 3.0, 18.0);
        let mut signal2 = SinSignal::new(0.1, 2.0, 10.0);
        let data1 = signal1.by_ref().take(200).collect::<Vec<(f64, f64)>>();
        let data2 = signal2.by_ref().take(200).collect::<Vec<(f64, f64)>>();
        Self {
            signal1,
            data1,
            signal2,
            data2,
            window: [0.0, 20.0],
            line_data: Vec::new(),
            stdin_rx: stdin_channel(),
            args,
        }
    }

    fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(250);
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
                self.on_tick();
                last_tick = Instant::now();
            }
        }
    }

    fn process_stdin_data(&mut self) {
        // we only care about the successful case
        // we want to keep the graphs up even if the input channel died
        // TODO: implement a popup + some status line to inform the user that no new data will arrive on disconnect
        while let Ok(msg) = self.stdin_rx.try_recv() {
            let mut splits = msg.split_whitespace();

            let get_f64 = |x: &mut SplitWhitespace| x.next().and_then(|x| x.parse::<f64>().ok());

            let (a, b) = (get_f64(&mut splits), get_f64(&mut splits));

            if let (Some(a), Some(b)) = (a, b) {
                debug!("pushing ({a:?}, {b:?})");
                self.line_data.push((a, b));
            }
        }

        if self
            .args
            .max_points
            .is_some_and(|max_points| self.line_data.len() > max_points)
        {
            self.line_data
                .drain(..(self.line_data.len() - self.args.max_points.unwrap()));
        }
    }

    fn on_tick(&mut self) {
        self.data1.drain(0..5);
        self.data1.extend(self.signal1.by_ref().take(5));

        self.data2.drain(0..10);
        self.data2.extend(self.signal2.by_ref().take(10));

        self.window[0] += 1.0;
        self.window[1] += 1.0;
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

        let graphs: Vec<Box<dyn FnMut(Rect, &mut Frame)>> = vec![
            Box::new(|panel, frame| {
                self.render_animated_chart(frame, panel);
            }),
            Box::new(|panel, frame| {
                render_barchart(frame, panel);
            }),
            Box::new(|panel, frame| {
                render_line_chart(frame, panel, &self.line_data);
            }),
            Box::new(|panel, frame| {
                render_scatter(frame, panel);
            }),
        ];

        for (panel, mut graph_fn) in panels.iter().zip(graphs) {
            graph_fn(*panel, frame);
        }
    }

    fn render_animated_chart(&self, frame: &mut Frame, area: Rect) {
        let x_labels = vec![
            Span::styled(
                format!("{}", self.window[0]),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{}", (self.window[0] + self.window[1]) / 2.0)),
            Span::styled(
                format!("{}", self.window[1]),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ];
        let datasets = vec![
            Dataset::default()
                .name("data2")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Cyan))
                .data(&self.data1),
            Dataset::default()
                .name("data3")
                .marker(symbols::Marker::Braille)
                .style(Style::default().fg(Color::Yellow))
                .data(&self.data2),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered())
            .x_axis(
                Axis::default()
                    .title("X Axis")
                    .style(Style::default().fg(Color::Gray))
                    .labels(x_labels)
                    .bounds(self.window),
            )
            .y_axis(
                Axis::default()
                    .title("Y Axis")
                    .style(Style::default().fg(Color::Gray))
                    .labels(["-20".bold(), "0".into(), "20".bold()])
                    .bounds([-20.0, 20.0]),
            );

        frame.render_widget(chart, area);
    }
}

fn render_barchart(frame: &mut Frame, bar_chart: Rect) {
    let dataset = Dataset::default()
        .marker(symbols::Marker::HalfBlock)
        .style(Style::new().fg(Color::Blue))
        .graph_type(GraphType::Bar)
        // a bell curve
        .data(&[
            (0., 0.4),
            (10., 2.9),
            (20., 13.5),
            (30., 41.1),
            (40., 80.1),
            (50., 100.0),
            (60., 80.1),
            (70., 41.1),
            (80., 13.5),
            (90., 2.9),
            (100., 0.4),
        ]);

    let chart = Chart::new(vec![dataset])
        .block(Block::bordered().title_top(Line::from("Bar chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .style(Style::default().gray())
                .bounds([0.0, 100.0])
                .labels(["0".bold(), "50".into(), "100.0".bold()]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().gray())
                .bounds([0.0, 100.0])
                .labels(["0".bold(), "50".into(), "100.0".bold()]),
        )
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

    frame.render_widget(chart, bar_chart);
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

fn render_line_chart(frame: &mut Frame, area: Rect, data: &[(f64, f64)]) {
    debug!("data to draw is: {data:?}");

    if data.is_empty() {
        return;
    }

    let min_max_x = get_min_max(data.iter().map(|(x, _)| x));
    let min_max_y = get_min_max(data.iter().map(|(_, y)| y));

    let bounds = Bounds {
        x: min_max_x.unwrap(),
        y: min_max_y.unwrap(),
    };

    let datasets = vec![Dataset::default()
        .name("Line from only 2 points".italic())
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Yellow))
        .graph_type(GraphType::Line)
        .data(data)];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(Line::from("Line chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().gray())
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

fn render_scatter(frame: &mut Frame, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Heavy")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::new().yellow())
            .data(&HEAVY_PAYLOAD_DATA),
        Dataset::default()
            .name("Medium".underlined())
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Style::new().magenta())
            .data(&MEDIUM_PAYLOAD_DATA),
        Dataset::default()
            .name("Small")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::new().cyan())
            .data(&SMALL_PAYLOAD_DATA),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(Line::from("Scatter chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .title("Year")
                .bounds([1960., 2020.])
                .style(Style::default().fg(Color::Gray))
                .labels(["1960", "1990", "2020"]),
        )
        .y_axis(
            Axis::default()
                .title("Cost")
                .bounds([0., 75000.])
                .style(Style::default().fg(Color::Gray))
                .labels(["0", "37 500", "75 000"]),
        )
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

fn update_state(
    state: &mut HashMap<GraphName, HashMap<LineName, Vec<DataPoint>>>,
    new_datum: &IntermediateLogRepresentation,
) -> Result<()> {
    let x = new_datum.x.parse::<f64>()?;

    let line_prefix = new_datum.line_tags.join("-");

    for graph in new_datum.graphs.iter() {
        // TODO: skip this extra lookup
        if !state.contains_key(graph) {
            state.insert(graph.clone(), HashMap::new());
        }

        let graph = state.get_mut(graph).unwrap();

        for (line, y) in new_datum.lines.iter() {
            let line_name = line_prefix.clone() + &line;

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

    /// parse all of the logs into the final graph structure
    #[test]
    fn test_parse_example_logs() {}

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

// Data from https://ourworldindata.org/space-exploration-satellites
const HEAVY_PAYLOAD_DATA: [(f64, f64); 9] = [
    (1965., 8200.),
    (1967., 5400.),
    (1981., 65400.),
    (1989., 30800.),
    (1997., 10200.),
    (2004., 11600.),
    (2014., 4500.),
    (2016., 7900.),
    (2018., 1500.),
];

const MEDIUM_PAYLOAD_DATA: [(f64, f64); 29] = [
    (1963., 29500.),
    (1964., 30600.),
    (1965., 177_900.),
    (1965., 21000.),
    (1966., 17900.),
    (1966., 8400.),
    (1975., 17500.),
    (1982., 8300.),
    (1985., 5100.),
    (1988., 18300.),
    (1990., 38800.),
    (1990., 9900.),
    (1991., 18700.),
    (1992., 9100.),
    (1994., 10500.),
    (1994., 8500.),
    (1994., 8700.),
    (1997., 6200.),
    (1999., 18000.),
    (1999., 7600.),
    (1999., 8900.),
    (1999., 9600.),
    (2000., 16000.),
    (2001., 10000.),
    (2002., 10400.),
    (2002., 8100.),
    (2010., 2600.),
    (2013., 13600.),
    (2017., 8000.),
];

const SMALL_PAYLOAD_DATA: [(f64, f64); 23] = [
    (1961., 118_500.),
    (1962., 14900.),
    (1975., 21400.),
    (1980., 32800.),
    (1988., 31100.),
    (1990., 41100.),
    (1993., 23600.),
    (1994., 20600.),
    (1994., 34600.),
    (1996., 50600.),
    (1997., 19200.),
    (1997., 45800.),
    (1998., 19100.),
    (2000., 73100.),
    (2003., 11200.),
    (2008., 12600.),
    (2010., 30500.),
    (2012., 20000.),
    (2013., 10600.),
    (2013., 34500.),
    (2015., 10600.),
    (2018., 23100.),
    (2019., 17300.),
];
