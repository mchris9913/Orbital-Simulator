use pyo3::prelude::*;

mod cubic_spline_1d {
	use pyo3::prelude::*;
	
	//Singular Point Structure for the spline
	struct Point {
		t: f64,
		x: Vec<f64>,
		k: Vec<f64>,
	}
	
	impl Point {
		pub fn new() -> Point {
			Point {
				t: 0.0,
				x: Vec::<f64>::new(),
				k: Vec::<f64>::new(),
			}
		}
		
		pub fn create(t: f64, x: Vec<f64>, k: Vec<f64>) -> Point {
			Point {
				t: t,
				x: x,
				k: k
			}
		}
	}
	
	//Spline Structure
	struct Spline {
		p1: Point,
		p2: Point,
		a: Vec<f64>,
		b: Vec<f64>,
	}
	
	impl Spline {
		pub fn new() -> Spline {
			Spline {
				p1: Point::new(),
				p2: Point::new(),
				a: Vec::<f64>::new(),
				b: Vec::<f64>::new(),
			}
		}
		
		pub fn update(&mut self, p1: Point, p2: Point) {
			self.p1 = p1;
			self.p2 = p2;
			self.a = vec![0.0; self.p1.x.len()];
			self.b = vec![0.0; self.p1.x.len()];
			
			for i in 0..self.p1.x.len() {
				self.a[i] = self.p1.k[i] * (self.p2.t - self.p1.t) - (self.p2.x[i] - self.p1.x[i]);
				self.b[i] = -self.p2.k[i] * (self.p2.t - self.p1.t) + (self.p2.x[i] - self.p1.x[i]);
			}
		}
		
		pub fn evaluate(&self, t: f64) -> Vec<f64> {
			let p: f64 = (t - self.p1.t)/(self.p2.t - self.p1.t);
			self.p1.x.iter()
			.zip(self.p2.x.iter())
			.zip(self.a.iter())
			.zip(self.b.iter())
			.map(|(((&i, &j), &k), &l)| (1. - p)*i + p*j + p*(1. - p)*((1. - p)*k + p*l))
			.collect()
		}
	}

	#[pyfunction]
	pub fn cubic_spline_1d_interpolate(t: Vec<f64>, data: (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)) -> Vec<Vec<f64>> {
		let mut out: Vec<Vec<f64>> = vec![vec![0.0; (data.1)[0].len()]; t.len()];
		
		let mut s: Spline = Spline::new();
		let mut update: bool = true;
		
		let mut j: usize = 0;
		for i in 0..t.len() {
			while j < data.0.len() - 1 {
				if data.0[j] == t[i] {
					out[i] = (data.1)[j].clone();
					break;
				}
				else if (data.0[j] < t[i]) && (data.0[j+1] > t[i]) {
					if update {
						s.update(Point::create(data.0[j],(data.1)[j].clone(),(data.2)[j].clone()), Point::create(data.0[j+1], (data.1)[j+1].clone(), (data.2)[j+1].clone()));
						update = false;
					}
					
					out[i] = s.evaluate(t[i]);
					break;
				}
				j += 1;
				update = true;
			}
		}
		
		out
	}
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
	m.add_function(wrap_pyfunction!(cubic_spline_1d::cubic_spline_1d_interpolate, m)?)?;
	Ok(())
}
