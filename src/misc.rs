use linked_hash_map::LinkedHashMap;
use std::hash::Hash;
use pyo3::{
    intern,
    sync::GILOnceCell,
    types::{PyAnyMethods, PyDict},
    Bound, IntoPyObject, PyAny, PyErr, PyObject, PyResult, Python,
};

pub fn clone_list<'py>(py: Python<'py>, list: &Vec<PyObject>) -> Vec<PyObject> {
    list.iter().map(|obj| obj.clone_ref(py)).collect()
}

pub fn tensor_slice_1d<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(tensor.call_method1(intern!(py, "narrow"), (0, start, stop - start))?)
}

pub fn torch_cat<'py>(py: Python<'py>, obj: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
    static INTERNED_CAT: GILOnceCell<PyObject> = GILOnceCell::new();
    Ok(INTERNED_CAT
        .get_or_try_init::<_, PyErr>(py, || Ok(py.import("torch")?.getattr("cat")?.unbind()))?
        .bind(py)
        .call1((obj,))?)
}

pub fn torch_empty<'py>(
    shape: &Bound<'py, PyAny>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    static INTERNED_EMPTY: GILOnceCell<PyObject> = GILOnceCell::new();
    let py = shape.py();
    Ok(INTERNED_EMPTY
        .get_or_try_init::<_, PyErr>(py, || Ok(py.import("torch")?.getattr("empty")?.unbind()))?
        .bind(py)
        .call(
            (shape,),
            Some(&PyDict::from_sequence(
                &vec![(intern!(py, "dtype"), dtype)].into_pyobject(py)?,
            )?),
        )?)
}

pub struct PyLinkedHashMap<K, V>(pub LinkedHashMap<K, V>);

impl<'py, K, V> pyo3::conversion::FromPyObject<'py> for PyLinkedHashMap<K, V>
where
    K: Eq + Hash + pyo3::conversion::FromPyObject<'py>,
    V: pyo3::conversion::FromPyObject<'py>,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let dict = ob.downcast::<PyDict>()?;
        let items = dict.call_method0(intern!(py, "items"))?;
        let list_builtin = py.import("builtins")?.getattr("list")?;
        let items_list = list_builtin.call1((items,))?;
        let kv_list: Vec<(K, V)> = items_list.extract()?;
        Ok(PyLinkedHashMap(kv_list.into_iter().collect()))
    }
}
