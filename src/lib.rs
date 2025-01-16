use pyo3::prelude::*;
use pyo3::Python;
use pyo3::exceptions::PyException;
use ndarray::prelude::*;

fn cubic_spline_interpolation_1d(t_interp: Array1<f64>, t: Array1<f64>, x: Array2<f64>, k: Array2<f64>) -> Array2<f64> {
	let num_interp_points = t_interp.raw_dim()[0];
	let num_points = t.raw_dim()[0];
	let num_states = x.raw_dim()[1];
	
	let mut t1: f64 = t[0];
	let mut x1 = x.slice(s![0, ..]);
	let mut k1 = k.slice(s![0, ..]);
	
	let mut t2: f64 = t[1];
	let mut x2 = x.slice(s![1, ..]);
	let mut k2 = k.slice(s![1, ..]);
	
	let mut a = &k1 * (t2 - t1) - (&x2 - &x1);
	let mut b = -1. * &k2 * (t2 - t1) + (&x2 - &x1);
	
	let mut i: usize = 0;
	let mut j: usize = 0;
	let mut update_parameters: bool = false;
	
	let mut out: Array2<f64> = Array::zeros((num_interp_points, num_states));
	
	while i < num_interp_points {
		let mut row = out.slice_mut(s![i, ..]);
		while j < num_points - 1 {
			if t_interp[i] == t[j] {
				let temp = x.slice(s![j, ..]); \\had to make this a seperate variable because it would error otherwise, not sure why
				row += &temp;
				break;
			} else if t_interp[i] > t[j] && t_interp[i] < t[j+1] {
				if update_parameters {
					update_parameters = false;
					t1 = t[j];
					x1 = x.slice(s![j, ..]);
					k1 = k.slice(s![j, ..]);
					
					t2 = t[j+1];
					x2 = x.slice(s![j+1, ..]);
					k2 = k.slice(s![j+1, ..]);
					
					a = &k1 * (t2 - t1) - (&x2 - &x1);
					b = -1. * &k2 * (t2 - t1) + (&x2 - &x1);
				}
				let p: f64 = (t_interp[i] - t1)/(t2 - t1);
				let interp = (1. - p) * &x1 + p * &x2 + p * (1. - p) * ((1. - p) * &a + p * &b); \\had to make this a seperate variable because if i did not then i would get an error not sure why
				row += &interp;
				break;
			}
			j += 1;
			update_parameters = true;
		}
		i += 1;
	}
	
	return out
}

#[pyfunction]
fn get_version() -> String {
	let version_number: &str = env!("CARGO_PKG_VERSION");
	let version_string: String = format!("{version_number}");
	return version_string
}

#[pymodule]
fn tiamat(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(get_version, m)?)?;
	Ok(())
}
