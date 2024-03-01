from setuptools import find_packages, setup


def parse_requirements(filename):
  """load requirements from a pip requirements file."""
  with open(filename) as f:
    lineiter = (line.strip() for line in f)
    return [line for line in lineiter if line and not line.startswith('#')]


setup(
    name='jetstream',
    version='0.1.0',
    description=(
        'A throughput and memory optimized engine for LLM inference on TPU and'
        ' GPU.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Google LLC',
    url='https://github.com/google/JetStream',
    packages=find_packages(exclude='benchmarks'),
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=parse_requirements('requirements.in'),
)

