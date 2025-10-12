use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::{
    collections::{BinaryHeap, HashSet},
    fmt::Debug,
};

use crate::utils::standard_deviation;

/// Internal heap element used in agglomerative clustering.
///
/// Each `HeapItem` stores two clusters (`i` and `j`) and their distance.
/// The [`BinaryHeap`] orders elements by increasing distance.
#[derive(PartialEq, Debug)]
struct HeapItem {
    /// Indices of the first cluster.
    i: Vec<usize>,
    /// Indices of the second cluster.
    j: Vec<usize>,
    /// Distance between clusters.
    distance: f64,
}

impl HeapItem {
    /// Checks if both clusters referenced by this heap item are still active.
    ///
    /// # Arguments
    ///
    /// * `active_set` - The set of currently active clusters.
    ///
    /// # Returns
    ///
    /// `true` if both `i` and `j` are active clusters; `false` otherwise.
    fn is_active(&self, active_set: &HashSet<Vec<usize>>) -> bool {
        active_set.contains(&self.i) && active_set.contains(&self.j)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse comparison for min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

impl Eq for HeapItem {}

/// Defines a distance function between two data elements.
///
/// This trait establishes the *metric space* structure for the clustering algorithms.
/// Any type implementing this trait can be used with [`Sample`] and [`ClusterCompare`].
///
/// # Example
///
/// ```rust
/// use larch::prelude::*;
///
/// #[derive(Clone, Copy)]
/// struct Point(f64, f64);
///
/// impl Metric<Point> for Point {
///     fn distance(&self, other: &Point) -> f64 {
///         ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
///     }
/// }
/// ```
pub trait Metric<D> {
    /// Computes the distance between `self` and another element.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compute distance to.
    ///
    /// # Returns
    ///
    /// A non-negative distance value between `self` and `other`.
    fn distance(&self, other: &D) -> f64;
}

/// Selects `k` elements from the dataset using low-density sampling.
///
/// # Arguments
///
/// * `data` - The dataset from which to sample.
/// * `k` - Number of samples to select.
///
/// # Returns
///
/// A vector containing `k` sampled elements.
///
/// # Behavior
///
/// - Starts from one random element.
/// - Each subsequent element is chosen with probability proportional
///   to the squared distance from existing samples.
pub fn low_density_sample<D>(data: &Vec<D>, k: usize) -> Vec<D>
where
    D: Metric<D> + Debug + Copy,
{
    let mut rng = rand::thread_rng();
    let mut sample = vec![];

    // Start with one random element
    sample.push(data[rng.gen_range(0..data.len())]);

    // Choose remaining samples based on distance weights
    for _ in 1..k {
        let weights = data
            .iter()
            .map(|d| {
                sample
                    .iter()
                    .map(|s| D::distance(d, s).powf(2.))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect::<Vec<_>>();

        let dist = WeightedIndex::new(&weights).unwrap();
        sample.push(data[dist.sample(&mut rng)]);
    }

    sample
}

/// Provides clustering operations for centroid-based algorithms.
///
/// This trait defines methods for k-means, hierarchical clustering,
/// and hybrid seeding strategies. Implementers define how to compute
/// a cluster’s centroid.
pub trait ClusterCompare<D> {
    /// Threshold multiplier controlling early stopping in agglomerative clustering.
    const BREAK_THRESHOLD: f64 = 1.5;

    /// Maximum number of k-means iterations.
    const MAX_ITERS: i32 = 10 ^ 4;

    /// Number of times to repeat k-means with different seeds.
    const KMEANS_REPEAT: usize = 5;

    /// Computes the centroid of a given cluster.
    ///
    /// # Arguments
    ///
    /// * `cluster` - The points forming the cluster.
    ///
    /// # Returns
    ///
    /// The computed centroid of type `D`.
    fn compute_centroid(cluster: &Vec<D>) -> D;

    /// Runs iterative k-means clustering until convergence or iteration limit.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to cluster.
    /// * `centroids` - Initial centroid positions.
    /// * `k` - Number of clusters.
    ///
    /// # Returns
    ///
    /// A vector of clusters, each containing indices of assigned data points.
    fn iterate_kmeans(data: &Vec<D>, centroids: Vec<D>, k: usize) -> Vec<Vec<usize>>
    where
        D: Metric<D> + ClusterCompare<D> + Copy,
    {
        let mut centroids = centroids;
        let mut clusters: Vec<Vec<usize>> = vec![];

        for _ in 0..Self::MAX_ITERS {
            let mut inner_clusters: Vec<Vec<usize>> = vec![vec![]; k];

            // Assign points to nearest centroid
            for i in 0..data.len() {
                let mut dist = vec![];
                for j in 0..k {
                    dist.push(D::distance(&data[i], &centroids[j]));
                }

                let mut best = (f64::MAX, 0);
                for (idx, &d) in dist.iter().enumerate() {
                    if d < best.0 {
                        best = (d, idx);
                    }
                }
                inner_clusters[best.1].push(i);
            }

            let temp_centroids = D::compute_centroids(data, &inner_clusters);

            // Check for convergence
            if temp_centroids
                .iter()
                .zip(centroids.iter())
                .map(|(c1, c2)| D::distance(c1, c2))
                .all(|d| d < f64::EPSILON)
            {
                break;
            }

            clusters = inner_clusters;
            centroids = temp_centroids;
        }

        clusters
    }

    /// Computes centroids for a collection of clusters.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset of points.
    /// * `clusters` - A vector of clusters, each holding indices into `data`.
    ///
    /// # Returns
    ///
    /// A vector of computed centroids.
    fn compute_centroids(data: &Vec<D>, clusters: &Vec<Vec<usize>>) -> Vec<D>
    where
        D: Metric<D> + ClusterCompare<D> + Copy,
    {
        clusters
            .iter()
            .filter(|c| !c.is_empty())
            .map(|c| D::compute_centroid(&c.iter().map(|&i| data[i]).collect::<Vec<_>>()))
            .collect()
    }

    /// Runs the standard k-means clustering algorithm using low-density sampling for
    /// initalizing centroids.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to cluster.
    /// * `k` - Number of clusters.
    ///
    /// # Returns
    ///
    /// A vector of clusters represented by index lists.
    fn kmeans(data: &Vec<D>, k: usize) -> Vec<Vec<usize>>
    where
        D: ClusterCompare<D> + Metric<D> + Copy + Debug,
    {
        let centroids = low_density_sample::<D>(data, k);
        D::iterate_kmeans(data, centroids, k)
    }

    /// Runs k-means using centroids derived from hierarchical clustering output.
    ///
    /// # Arguments
    ///
    /// * `data` - Full dataset to recluster.
    /// * `clusters` - Hierarchical clusters to derive seeds from.
    ///
    /// # Returns
    ///
    /// Refined clusters after seeded k-means.
    fn hierarchical_seeded_kmeans(data: &Vec<D>) -> Vec<Vec<usize>>
    where
        D: ClusterCompare<D> + Metric<D> + Copy,
    {
        let clusters = D::agglomerative_cluster(data);
        let centroids = D::compute_centroids(data, &clusters);
        let k = clusters.len();
        D::iterate_kmeans(data, centroids, k)
    }

    /// Runs k-means using centroids computed from a *sample* of the dataset.
    ///
    /// Useful for large datasets where full hierarchical seeding is costly.
    ///
    /// # Arguments
    ///
    /// * `data` - Full dataset.
    /// * `sample` - Subsample used to compute initial centroids.
    /// * `clusters` - Clusters computed on the sample.
    ///
    /// # Returns
    ///
    /// Final clusters over the full dataset.
    fn hierarchical_seeded_kmeans_from_sample(
        data: &Vec<D>,
        sample: &Vec<D>,
        clusters: &Vec<Vec<usize>>,
    ) -> Vec<Vec<usize>>
    where
        D: ClusterCompare<D> + Metric<D> + Copy,
    {
        let centroids = D::compute_centroids(sample, clusters);
        let k = clusters.len();
        D::iterate_kmeans(data, centroids, k)
    }

    /// Performs agglomerative (hierarchical) clustering with average-linkage updates.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to cluster.
    ///
    /// # Returns
    ///
    /// A vector of clusters, each represented as a list of data indices.
    ///
    /// # Behavior
    ///
    /// - Initializes all elements as singleton clusters.
    /// - Iteratively merges the closest pair of clusters.
    /// - Stops when the minimum distance exceeds `BREAK_THRESHOLD × σ`,
    ///   where `σ` is the standard deviation of all pairwise distances.
    fn agglomerative_cluster(data: &Vec<D>) -> Vec<Vec<usize>>
    where
        D: ClusterCompare<D> + Metric<D> + Copy,
    {
        let mut active: HashSet<Vec<usize>> = (0..data.len()).map(|i| vec![i]).collect();

        let mut vals = vec![];
        let mut distance_heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        let mut distance_matrix = vec![vec![0.0; data.len()]; data.len()];

        // Initialize pairwise distances
        for i in 0..data.len() + 1 {
            for j in i + 1..data.len() {
                let d = D::distance(&data[i], &data[j]);
                distance_matrix[i][j] = d;
                distance_matrix[j][i] = d;
                distance_heap.push(HeapItem {
                    i: vec![i],
                    j: vec![j],
                    distance: d,
                });
                vals.push(d);
            }
        }

        let std = standard_deviation(&vals).unwrap();

        // Merge clusters until threshold
        for _ in 0..data.len() - 1 {
            let mut min_item = distance_heap.pop().unwrap();
            while !min_item.is_active(&active) {
                min_item = distance_heap.pop().unwrap();
            }

            if Self::BREAK_THRESHOLD * std < min_item.distance {
                break;
            }

            active.remove(&min_item.i);
            active.remove(&min_item.j);

            let mut new_active = min_item.i.clone();
            new_active.append(&mut min_item.j.clone());

            // Update average distances to new cluster
            for active_v in &active {
                let m_idx = active_v[0];
                let dist = (distance_matrix[min_item.i[0]][m_idx]
                    + distance_matrix[min_item.j[0]][m_idx])
                    / 2.0;

                distance_matrix[min_item.i[0]][m_idx] = dist;
                distance_heap.push(HeapItem {
                    i: new_active.clone(),
                    j: active_v.clone(),
                    distance: dist,
                });
            }

            active.insert(new_active.clone());
        }

        active.into_iter().collect::<Vec<_>>()
    }
}
