from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="SMdRQA",
    version="2024.03.22",
    author="Swarag Thaikkandi, K.M. Sharika, Miss Nivedita",
    author_email="tsk4at@gmail.com",
    description="A Python implementation of MdRQA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SwaragThaikkandi/Sliding_MdRQA",
    packages=find_packages(),
    install_requires=['numpy','scipy','operator-courier','contextlib2','pytest-warnings','matplotlib','pandas','tqdm','memory_profiler','kuramoto','networkx','p_tqdm'],
    setup_requires=['numpy','scipy','operator-courier','contextlib2','pytest-warnings','matplotlib','pandas','tqdm','memory_profiler','kuramoto','networkx','p_tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

