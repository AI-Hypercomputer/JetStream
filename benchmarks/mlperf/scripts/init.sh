git clone https://github.com/google/jax.git
cd jax
git reset 44359cb30ab5cdbe97e6b78c2c64fe9f8add29ca --hard
pip install -e .
gsutil cp gs://zhihaoshan-maxtext-profiling/jax_proxy_stream_buffer/jaxlib-0.4.31.dev20240719-cp310-cp310-manylinux2014_x86_64-mlperf_version_3.whl .
mv jaxlib-0.4.31.dev20240719-cp310-cp310-manylinux2014_x86_64-mlperf_version_3.whl jaxlib-0.4.31.dev20240719-cp310-cp310-manylinux2014_x86_64.whl
pip install jaxlib-0.4.31.dev20240719-cp310-cp310-manylinux2014_x86_64.whl
