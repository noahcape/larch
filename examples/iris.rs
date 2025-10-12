use larch::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = vec![];
    let mut rdr = Reader::from_path("./examples/iris.csv")?;
    for result in rdr.deserialize() {
        let record: Iris = result?;
        data.push(record);
    }

    let clusters = Iris::hierarchical_seeded_kmeans(&data);

    println!("Hierarchical Seeding kmeans");
    for cluster in clusters {
        let setosa = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Setosa)
            .collect::<Vec<_>>();
        let versicolor = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Versicolor)
            .collect::<Vec<_>>();
        let virginica = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Virginica)
            .collect::<Vec<_>>();

        println!(
            "Cluster: setosa: {} versicolor: {} virginica: {}",
            setosa.len(),
            versicolor.len(),
            virginica.len()
        );
    }

    let clusters = Iris::kmeans(&data, 3);

    println!("kmeans++");
    for cluster in clusters {
        let setosa = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Setosa)
            .collect::<Vec<_>>();
        let versicolor = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Versicolor)
            .collect::<Vec<_>>();
        let virginica = cluster
            .iter()
            .map(|&i| data[i])
            .filter(|&i| i.species == IrisType::Virginica)
            .collect::<Vec<_>>();

        println!(
            "Cluster: setosa: {} versicolor: {} virginica: {}",
            setosa.len(),
            versicolor.len(),
            virginica.len()
        );
    }

    Ok(())
}

use csv::Reader;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
enum IrisType {
    #[serde(rename = "setosa")]
    Setosa,
    #[serde(rename = "versicolor")]
    Versicolor,
    #[serde(rename = "virginica")]
    Virginica,
    None,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: IrisType,
}

impl Metric<Iris> for Iris {
    fn distance(&self, other: &Iris) -> f64 {
        let diffs = vec![
            self.sepal_width - other.sepal_width,
            self.sepal_width - other.sepal_width,
            self.petal_length - other.petal_length,
            self.petal_width - other.petal_width,
        ];

        diffs
            .iter()
            .fold(0., |acc, &diff| acc + diff.powf(2.))
            .sqrt()
    }
}

impl ClusterCompare<Iris> for Iris {
    fn compute_centroid(data: &Vec<Iris>) -> Iris {
        let mut sl = 0.;
        let mut sw = 0.;
        let mut pl = 0.;
        let mut pw = 0.;

        for p in data {
            sl += p.sepal_length;
            sw += p.sepal_width;
            pl += p.petal_length;
            pw += p.petal_width;
        }

        Iris {
            sepal_length: sl / data.len() as f64,
            sepal_width: sw / data.len() as f64,
            petal_length: pl / data.len() as f64,
            petal_width: pw / data.len() as f64,
            species: IrisType::None,
        }
    }
}
