use plotters::{
    prelude::*,
    style::full_palette::{BROWN, ORANGE, PURPLE, TEAL},
};
use std::fmt::Debug;

use larch::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Metric<Point> for Point {
    fn distance(&self, other: &Point) -> f64 {
        (((self.x - other.x).pow(2) + (self.y - other.y).pow(2)) as f64).sqrt()
    }
}

impl ClusterCompare<Point> for Point {
    fn compute_centroid(data: &Vec<Point>) -> Point {
        let mut x = 0;
        let mut y = 0;

        for p in data {
            x += p.x;
            y += p.y;
        }

        Point {
            x: x / data.len() as i32,
            y: y / data.len() as i32,
        }
    }
}

fn main() {
    use rand::Rng;

    let mut data = vec![];

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        data.push(Point {
            x: rng.gen_range(0..1000),
            y: rng.gen_range(0..1000),
        })
    }

    let clusters = Point::kmeans(&data, 5);
    let _ = visualize(
        &data,
        &clusters,
        1000,
        "./examples/RAND_kmeans_cluster_rand.png",
    );

    let clusters = Point::hierarchical_seeded_kmeans(&data);
    let _ = visualize(
        &data,
        &clusters,
        1000,
        "./examples/RAND_seeded_kmeans_cluster_rand.png",
    );

    let sample = low_density_sample::<Point>(&data, (data.len() as f64 * 0.1).ceil() as usize);
    let clusters = Point::agglomerative_cluster(&sample);
    let clusters = Point::hierarchical_seeded_kmeans_from_sample(&data, &sample, &clusters);
    let _ = visualize(
        &data,
        &clusters,
        1000,
        "./examples/RAND_seeded_kmeans_cluster_rand_from_sample.png",
    );
}

fn visualize(
    data: &Vec<Point>,
    clusters: &Vec<Vec<usize>>,
    scale: i32,
    fname: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(fname, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Clustered Points", ("sans-serif", 30))
        .set_all_label_area_size(40)
        .build_cartesian_2d(-10..scale, -10..scale)?; // axis ranges

    chart.configure_mesh().draw()?;

    let k = clusters
        .iter()
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>()
        .len();

    if k > 10 {
        panic!("Can't plot more than 10 colors")
    }

    let palette = match k {
        1 => vec![&RED],
        2 => vec![&RED, &BLUE],
        3 => vec![&RED, &BLUE, &GREEN],
        4 => vec![&RED, &BLUE, &GREEN, &MAGENTA],
        5 => vec![&RED, &BLUE, &GREEN, &MAGENTA, &BLACK],
        6 => vec![&RED, &BLUE, &GREEN, &MAGENTA, &BLACK, &PURPLE],
        7 => vec![&RED, &BLUE, &GREEN, &MAGENTA, &BLACK, &PURPLE, &YELLOW],
        8 => vec![
            &RED, &BLUE, &GREEN, &MAGENTA, &BLACK, &PURPLE, &YELLOW, &ORANGE,
        ],
        9 => vec![
            &RED, &BLUE, &GREEN, &MAGENTA, &BLACK, &PURPLE, &YELLOW, &ORANGE, &BROWN,
        ],
        10 => vec![
            &RED, &BLUE, &GREEN, &MAGENTA, &BLACK, &PURPLE, &YELLOW, &ORANGE, &BROWN, &TEAL,
        ],
        _ => vec![],
    };

    for (cluster_idx, cluster) in clusters.iter().filter(|v| !v.is_empty()).enumerate() {
        let color = palette[cluster_idx % palette.len()];
        for &idx in cluster {
            let p = data[idx];
            chart.draw_series(std::iter::once(Circle::new((p.x, p.y), 5, color.filled())))?;
        }
    }

    Ok(())
}
