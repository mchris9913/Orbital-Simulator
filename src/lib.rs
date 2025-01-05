use pyo3::prelude::*;
use pyo3::Python;
use pyo3::exceptions::PyException;

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