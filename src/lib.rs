//! # larch
//!
//! This crate provides a generic, trait-based framework for clustering and sampling algorithms.
//! The main goal of this library is to implemenent a hierarchical clustering method to determine
//! k for seeding a kmeans clustering algorithm.
//!
//! It supports:
//! - Metric-based clustering via the [`Metric`] trait.
//! - Hierachical seeding of kmeans through tree cutting agglomerative clustering algorithm.
//! - Centroid-based clustering methods like *k-means* and *hierarchical clustering* through [`ClusterCompare`].
//!
//! ## Overview
//!
//! The crate is designed for composability and generic use. Implement the [`Metric`] trait
//! for your own data type, then derive additional clustering behavior automatically through
//! the provided default methods.
//!
//! ## Example
//!
//! ```rust
//! use larch::prelude::*;
//!
//! #[derive(Clone, Copy, Debug)]
//! struct Point(f64, f64);
//!
//! impl Metric<Point> for Point {
//!     fn distance(&self, other: &Point) -> f64 {
//!         ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
//!     }
//! }
//!
//! impl ClusterCompare<Point> for Point {
//!     fn compute_centroid(data: &Vec<Point>) -> Point {
//!         let (sx, sy): (f64, f64) = data.iter().fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
//!         let n = data.len() as f64;
//!         Point(sx / n, sy / n)
//!     }
//! }
//!
//!
//! let points = vec![Point(0.0, 0.0), Point(1.0, 1.0), Point(10.0, 10.0)];
//! // normal k-means
//! let kmeans_clusters = Point::kmeans(&points, 2);
//! // determine k by hierarchical clustering + tree cutting then run kmeans
//! let seeded_clusters = Point::hierarchical_seeded_kmeans(&points);
//! println!("K-means clusters: {:?}", kmeans_clusters);
//! println!("Seeded K-means clusters: {:?}", seeded_clusters);
//! ```
//! ## Note
//! This library is not highly optmized and due to the Rust implementation of [`BinaryHeap`] the hierarchical
//! clustering is not deterministic.
//!
//! ## License
//! larch is licensed under the BSD 3 license (see `LICENSE` in the main repository).
pub mod cluster;
mod utils;

pub mod prelude {
    pub use super::cluster::*;
}
