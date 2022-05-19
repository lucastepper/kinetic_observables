use pyo3::prelude::*;
use pyo3 as py;
use ndarray as nd;
use numpy as np;


// convert PyAny into ndarray if PyAny is List or ndarray
fn extract_list_array(to_extract: &PyAny, name: &str) -> nd::Array1<f64> {
    let pytype = to_extract.get_type().name().unwrap();
    match pytype {
        "list" => nd::Array1::from_vec(to_extract.extract::<Vec<f64>>().unwrap()),
        "ndarray" => {
            let array = to_extract.extract::<&np::PyArray1<f64>>().unwrap();
            unsafe { array.as_array().to_owned() }
        },
        _ => panic!("{} must be a list or ndarray", name),
    }
}

#[pyclass]
struct PassageTimes {
    #[pyo3(get, set)]
    dt: f64,
    starts: nd::Array1<f64>,
    ends: nd::Array1<f64>,
}
#[pymethods]
impl PassageTimes {
    #[new]
    fn new<'py> (dt: f64, starts: &'py py::types::PyAny, ends: &'py py::types::PyAny) -> Self {
        let starts = extract_list_array(starts, "starts");
        let ends = extract_list_array(ends, "ends");
        PassageTimes {
            dt,
            starts,
            ends,
        }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("PassageTimes(dt={}, starts={}, ends={})", &self.dt, &self.starts, &self.ends))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("PassageTimes(dt={}, starts={}, ends={})", &self.dt, &self.starts, &self.ends))
    }

}


#[pymodule]
fn kin_obs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PassageTimes>()?;
    Ok(())
}
