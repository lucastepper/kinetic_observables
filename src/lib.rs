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

fn get_fpt_single(traj: nd::ArrayView1<f64>, start: f64, end: f64) -> (f64, usize) {
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

fn get_fpts(
    traj: nd::ArrayView1<f64>,
    starts: &nd::Array1<f64>,
    ends: &nd::Array1<f64>,
    dt: f64,
) -> (nd::Array2<f64>, nd::Array2<usize>) {
    let mut fpts_sum: nd::Array2<f64> = nd::Array2::zeros((starts.len(), ends.len()));
    let mut counter: nd::Array2<usize> = nd::Array2::zeros((starts.len(), ends.len()));
    for i in 0..starts.len() {
        for j in 0..ends.len() {
            let (fpt_sum, count) = get_fpt_single(traj, starts[i], ends[j]);
            fpts_sum[(i, j)] += dt * fpt_sum;
            counter[(i, j)] += count;
        }
    }
    (fpts_sum, counter)
}

// convert PyAny into ndarray if PyAny is List or ndarray
fn extract_list_array(to_extract: &PyAny, name: &str) -> nd::Array1<f64> {
    let pytype = to_extract.get_type().name().unwrap();
    match pytype {
        "list" => nd::Array1::from_vec(to_extract.extract::<Vec<f64>>().unwrap()),
        "ndarray" => {
            let array = to_extract.extract::<&np::PyArray1<f64>>().unwrap();
            unsafe { array.as_array().to_owned() }
        }
        _ => panic!("{} must be a list or ndarray", name),
    }
}

#[pyclass(text_signature="(starts, ends, dt)")]
struct PassageTimes {
    #[pyo3(get, set)]
    dt: f64,
    starts: nd::Array1<f64>,
    ends: nd::Array1<f64>,
    fpts_sum: nd::Array2<f64>,
    fpts_counter: nd::Array2<usize>,
}
#[pymethods]
impl PassageTimes {
    #[new]
    fn new<'py>(starts: &'py py::types::PyAny, ends: &'py py::types::PyAny, dt: f64) -> Self {
        let starts = extract_list_array(starts, "starts");
        let ends = extract_list_array(ends, "ends");
        let fpts_sum: nd::Array2<f64> = nd::Array2::zeros((starts.len(), ends.len()));
        let fpts_counter: nd::Array2<usize> = nd::Array2::zeros((starts.len(), ends.len()));
        PassageTimes {
            dt,
            starts,
            ends,
            fpts_sum,
            fpts_counter,
        }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "PassageTimes(dt={}, starts={}, ends={})",
            &self.dt, &self.starts, &self.ends
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PassageTimes(dt={}, starts={}, ends={}, fpts_sum={:?}, fpts_counter={:?})",
            &self.dt,
            &self.starts,
            &self.ends,
            &self.fpts_sum.as_slice().unwrap(),
            &self.fpts_counter.as_slice().unwrap()
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
        let mut trajs = unsafe { trajs.as_array_mut() };
        let mut results = Vec::new();
        if trajs.shape().len() == 1 {
            // compute results in serial if there is only one trajectory
            let traj =
                nd::ArrayView1::from_shape(trajs.len(), trajs.as_slice_mut().unwrap()).unwrap();
            results.push(get_fpts(traj, &self.starts, &self.ends, self.dt));
        } else if trajs.shape().len() == 2 {
            // compute results in parallel over trajs
            trajs
                .axis_iter(nd::Axis(0))
                .into_par_iter()
                .map(|row| {
                    let traj =
                        nd::ArrayView1::from_shape(row.len(), row.as_slice().unwrap()).unwrap();
                    get_fpts(traj, &self.starts, &self.ends, self.dt)
                })
                .collect_into_vec(&mut results);
        } else {
            return Err(PyErr::new::<py::exceptions::PyValueError, _>(
                "trajs must have ndims == 1 or 2",
            ));
        }
        // add results
        for (sum, counter) in &results {
            self.fpts_sum += sum;
            self.fpts_counter += counter;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    fn get_result<'py>(&self, py: Python<'py>) -> PyResult<&'py np::PyArray2<f64>> {
        let result = &self.fpts_sum / self.fpts_counter.mapv(|x| x as f64);
        Ok(result.into_pyarray(py))
    }
}

#[pymodule]
fn kin_obs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PassageTimes>()?;
    Ok(())
}
