use ndarray as nd;
use ndarray::parallel::prelude::*;
use numpy as np;
use numpy::IntoPyArray;
use pyo3 as py;
use pyo3::prelude::*;

fn is_transition(x1: f64, x2: f64, barrier: f64) -> bool {
    if x1 <= barrier && x2 > barrier {
        return true;
    }
    if x1 > barrier && x2 <= barrier {
        return true;
    }
    false
}

// Get the index of the bin in which x falls, 0 if x is left of start, nbins if x is right of end
// Example diagram for nbins = 2 and start = 0, end = 1;
//  idx=0 |boundary=0.0| idx=1 |boundary=1.0| idx=2=nbins
fn get_idx(x: f64, start: f64, end: f64, dx: f64, nbins: i16) -> i16 {
    if x < start {
        0
    } else if x >= end {
        nbins
    } else {
        ((x - start) / dx) as i16 + 1
    }
}

/// To facilitate an easier computation of the passage times, we first run through the trajectory
/// and find all transitions over the boundaries. For this, we first give each frame an index,
/// where the part left of the first bondary is 0, and hence the part right of the last boundary
/// is nbins+1. Then we go through all the pairs of indices and check if they are transitions.
/// If we find that the indicies change between two frames, we have a transition and at the time
/// index at which it happens and over which boundary it happens to our results. If we find that
/// more than one boundary is crossed, we just add two entries with the same time index
fn get_transitions(
    traj: nd::ArrayView1<f64>,
    start: f64,
    end: f64,
    nbins: usize,
) -> PyResult<(Vec<usize>, Vec<i16>)> {
    if nbins < 2 {
        return Err(PyErr::new::<py::exceptions::PyValueError, _>(
            "nbins must be >= 2",
        ));
    }
    if nbins > 500 {
        return Err(PyErr::new::<py::exceptions::PyValueError, _>(
            "nbins must be <= 500",
        ));
    }
    let nbins = nbins as i16;
    let dx = (end - start).abs() as f64 / (nbins as f64);
    // println!("dx: {}", dx);
    // Lets guess that we need sqrt(len(traj)) entries
    let mut times = Vec::with_capacity((traj.len() as f64).sqrt() as usize);
    let mut idxs = Vec::with_capacity((traj.len() as f64).sqrt() as usize);
    let mut idx_frame_last = get_idx(traj[0], start, end, dx, nbins);
    for iframe in 1..traj.len() {
        let idx_frame = get_idx(traj[iframe], start, end, dx, nbins);
        // We label the transition always by its right boundary time index, for mathematical
        // convenience. In the end, which time index we choose does not matter, as long as
        // we are consistent.
        // To find the right boundary index, if we are transitioning from left to right, we
        // need to take our previous index, and if we are transitioning from right to left, we
        // need to substract one from our current index.
        // println!(
        //     "iframe {} idx_frame_last: {} for frame last {}, idx_frame: {} for frame {}",
        //     iframe,
        //     idx_frame_last,
        //     traj[iframe - 1],
        //     idx_frame,
        //     traj[iframe]
        // );
        while idx_frame_last != idx_frame {
            // println!(
            //     "Detected transition as idx_frame_last {} != idx_frame {}",
            //     idx_frame_last, idx_frame
            // );
            // going to the right is transition_direction = 1
            let transition_direction = (idx_frame - idx_frame_last).signum();
            // println!("Detected transition direction {}", transition_direction);
            times.push(iframe);
            if transition_direction == -1 {
                // println!("Added transition idx {}", idx_frame_last - 1);
                idxs.push(idx_frame_last - 1);
            } else {
                // println!("Added transition idx {}", idx_frame_last);
                idxs.push(idx_frame_last);
            }
            idx_frame_last += transition_direction;
        }
    }
    Ok((times, idxs))
}

/// Wrapper function to get the first passage time to allow calling from python
/// from the test module
#[pyfunction(trajs, start, end, nbins)]
fn get_transition_times_idxs<'py>(
    trajs: &'py np::PyArray1<f64>,
    start: f64,
    end: f64,
    nbins: usize,
) -> PyResult<(Vec<usize>, Vec<i16>)> {
    let trajs_view = unsafe { trajs.as_array() };
    get_transitions(trajs_view, start, end, nbins)
}

fn get_fpt_single_old(traj: nd::ArrayView1<f64>, start: f64, end: f64) -> (f64, usize) {
    let mut count = 0;
    let mut fpt_sum = 0.;
    let mut end_found = usize::MAX;
    // iterate through from end, add all diffs from end found
    // to every start found, update end when found
    for mut i in 1..traj.len() {
        i = traj.len() - i - 1;
        if is_transition(traj[i as usize], traj[i + 1], end) {
            end_found = i;
        }
        if end_found < usize::MAX && is_transition(traj[i], traj[i + 1], start) {
            fpt_sum += (end_found - i) as f64;
            count += 1;
        }
    }
    (fpt_sum, count)
}

fn get_fpt_single(
    transition_times: Vec<usize>,
    transition_idxs: Vec<i16>,
    nbins: usize,
) -> (nd::Array2<f64>, nd::Array2<usize>) {
    let mut fpt_sums = nd::Array2::<f64>::zeros((2, nbins));
    let mut fpt_counts = nd::Array2::<usize>::zeros((2, nbins));
    let nbins_minus_one = (nbins - 1) as i16;
    // iterate through all transitions from end, then, for each transition over bin idx_j
    // iterate through all other bins idx_i, then, if idx_i == start or idx_i == end,
    // (We ignore all start points between (start, end) for numerical efficiency)
    // and add the difference in transition times
    // at fpt_sum[idx_i, idx_j] and increase the count at fpt_counts[idx_i, idx_j]
    // do this until we find a transition over bin idx_j, again
    for j in (1..transition_times.len()).rev() {
        let idx_j = transition_idxs[j];
        let time_j = transition_times[j];
        let mut i = j - 1;
        while i > 0 && transition_idxs[i] != idx_j {
            if (idx_j == 0 || idx_j == nbins_minus_one) {
                let idx_i = transition_idxs[i] as usize;
                let idx_j = idx_j as usize;
                fpt_sums[[0, 0]] += (time_j - transition_times[i]) as f64;
                fpt_counts[[idx_i, idx_j]] += 1;
            }
            i -= 1;
        }
    }
    (fpt_sums, fpt_counts)
}

fn get_ffpt_single_old(traj: nd::ArrayView1<f64>, start: f64, end: f64) -> (f64, usize) {
    let mut count = 0;
    let mut ffpt_sum = 0.;
    let mut start_found = usize::MAX;
    // iterate through from end, add all diffs from end found
    // to every start found, update end when found
    for i in 0..(traj.len() - 1) {
        if start_found < usize::MAX {
            if is_transition(traj[i], traj[i + 1], end) {
                ffpt_sum += (i - start_found) as f64;
                count += 1;
                start_found = usize::MAX;
            }
        } else {
            if is_transition(traj[i], traj[i + 1], start) {
                start_found = i;
            }
        }
    }
    (ffpt_sum, count)
}

fn get_passage_times_old(
    traj: nd::ArrayView1<f64>,
    starts: &nd::Array1<f64>,
    ends: &nd::Array1<f64>,
    dt: f64,
    method: &PassageTimesMethod,
) -> (nd::Array2<f64>, nd::Array2<usize>) {
    let mut passage_times_sum: nd::Array2<f64> = nd::Array2::zeros((starts.len(), ends.len()));
    let mut counter: nd::Array2<usize> = nd::Array2::zeros((starts.len(), ends.len()));
    for i in 0..starts.len() {
        for j in 0..ends.len() {
            let (fpt_sum, count) = match method {
                PassageTimesMethod::First => get_ffpt_single_old(traj, starts[i], ends[j]),
                PassageTimesMethod::All => get_fpt_single_old(traj, starts[i], ends[j]),
            };
            passage_times_sum[(i, j)] += dt * fpt_sum;
            counter[(i, j)] += count;
        }
    }
    (passage_times_sum, counter)
}

fn get_passage_times(
    traj: nd::ArrayView1<f64>,
    starts: &nd::Array1<f64>,
    ends: &nd::Array1<f64>,
    dt: f64,
    method: &PassageTimesMethod,
) -> (nd::Array2<f64>, nd::Array2<usize>) {
    let mut passage_times_sum: nd::Array2<f64> = nd::Array2::zeros((starts.len(), ends.len()));
    let mut counter: nd::Array2<usize> = nd::Array2::zeros((starts.len(), ends.len()));
    for i in 0..starts.len() {
        for j in 0..ends.len() {
            let (fpt_sum, count) = match method {
                PassageTimesMethod::First => get_ffpt_single_old(traj, starts[i], ends[j]),
                PassageTimesMethod::All => get_fpt_single_old(traj, starts[i], ends[j]),
            };
            passage_times_sum[(i, j)] += dt * fpt_sum;
            counter[(i, j)] += count;
        }
    }
    (passage_times_sum, counter)
}

// convert PyAny into ndarray if PyAny is List or ndarray
fn extract_list_array(to_extract: &PyAny, name: &str) -> nd::ArrayD<f64> {
    let pytype = to_extract.get_type().name().unwrap();
    match pytype {
        "list" => nd::Array1::from_vec(to_extract.extract::<Vec<f64>>().unwrap()).into_dyn(),
        "ndarray" => {
            let array = to_extract.extract::<&np::PyArrayDyn<f64>>().unwrap();
            unsafe { array.as_array().to_owned() }
        }
        _ => panic!("{} must be a list or ndarray", name),
    }
}

enum PassageTimesMethod {
    All,
    First,
}

#[pyclass(text_signature = "(starts, ends, dt, method)")]
struct PassageTimes {
    #[pyo3(get, set)]
    dt: f64,
    start: f64,
    end: f64,
    nbins: usize,
    passage_times_sum: nd::Array2<f64>,
    passage_times_counter: nd::Array2<usize>,
    method: PassageTimesMethod,
}
#[pymethods]
impl PassageTimes {
    #[new]
    fn new<'py>(start: f64, end: f64, nbins: usize, dt: f64, method: String) -> PyResult<Self> {
        let passage_times_sum: nd::Array2<f64> = nd::Array2::zeros((2, nbins));
        let passage_times_counter: nd::Array2<usize> = nd::Array2::zeros((2, nbins));
        let method = match method.as_str() {
            "all" => PassageTimesMethod::All,
            "first" => PassageTimesMethod::First,
            _ => {
                return Err(PyErr::new::<py::exceptions::PyValueError, _>(
                    "method must be either 'all' or 'first'",
                ))
            }
        };
        Ok(PassageTimes {
            dt,
            start,
            end,
            nbins,
            passage_times_sum,
            passage_times_counter,
            method,
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "PassageTimes(dt={}, starts={}, ends={})",
            &self.dt, &self.start, &self.end
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PassageTimes(dt={}, starts={}, ends={}, passage_times_sum={:?}, passage_times_counter={:?})",
            &self.dt,
            &self.start,
            &self.end,
            &self.passage_times_sum.as_slice().unwrap(),
            &self.passage_times_counter.as_slice().unwrap()
        ))
    }

    /// Add data iteratively to the object
    #[pyo3(text_signature = "($self, trajs)")]
    fn add_data<'py>(&mut self, trajs: &'py np::PyArrayDyn<f64>) -> PyResult<()> {
        if !trajs.is_c_contiguous() {
            return Err(PyErr::new::<py::exceptions::PyValueError, _>(
                "trajs must be c-contiguous",
            ));
        }
        // Make the code work for 1D and 2D arrays by reshaping 1D arrays to 2D
        // let mut results = Vec::new();
        let trajs_view = if trajs.ndim() == 1 {
            unsafe { trajs.reshape([1, trajs.len()]).unwrap().as_array() }
        } else if trajs.ndim() == 2 {
            // We do not change the shape here but convert it to fixed size array
            // implicitly telling the comlier that ndim == 2 by usingz size [n, m]
            unsafe {
                trajs
                    .reshape([trajs.shape()[0], trajs.shape()[1]])
                    .unwrap()
                    .as_array()
            }
        } else {
            return Err(PyErr::new::<py::exceptions::PyValueError, _>(format!(
                "trajs must have ndims == 1 or 2, got ndims = {}",
                trajs.ndim()
            )));
        };
        // let (times_transition, idxs_transition)
        for traj in trajs_view.axis_iter(nd::Axis(0)) {
            let (times_transition, idxs_transition) =
                get_transitions(traj, self.start, self.end, self.nbins)?;
            let (fpt_sums, counts) = get_fpt_single(times_transition, idxs_transition, self.nbins);
            self.passage_times_sum += &fpt_sums;
            self.passage_times_counter += &counts;
            // println!("Times transition {:?}", times_transition);
            // println!("IDXS transition {:?}", idxs_transition);
        }
        // compute results in parallel over trajs
        // trajs_view
        //     .axis_iter(nd::Axis(0))
        //     .into_par_iter()
        //     .map(|row| {
        //         let traj = nd::ArrayView1::from_shape(row.len(), row.as_slice().unwrap()).unwrap();
        //         get_passage_times_old(traj, &self.starts, &self.ends, self.dt, &self.method)
        //     })
        //     .collect_into_vec(&mut results);
        // add results
        // for (sum, counter) in &results {
        //     self.passage_times_sum += sum;
        //     self.passage_times_counter += counter;
        // }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    fn get_result<'py>(&self, py: Python<'py>) -> PyResult<&'py np::PyArray2<f64>> {
        let result = &self.passage_times_sum / self.passage_times_counter.mapv(|x| x as f64);
        Ok(result.into_pyarray(py))
    }
}

#[pymodule]
fn kin_obs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PassageTimes>()?;
    m.add_function(wrap_pyfunction!(get_transition_times_idxs, m)?)
        .unwrap();
    Ok(())
}
