# pyfastGEAR

**pyfastGEAR** is a Python reimplementation of **fastGEAR**, a method for detecting both recent and ancestral recombination events in bacterial populations.

* Original fastGEAR (MATLAB) software: https://users.ics.aalto.fi/pemartti/fastGEAR/
* Original publication: https://academic.oup.com/mbe/article/34/5/1167/2983515
* Original code, modified to run with Octave, a free and open source reimplementation of MATLAB is available at nzmacalasdair/fastGEAR-octave


## Status

⚠️ **pyfastGEAR is currently under active development.**

While the software has been tested extensively on a limited number of datasets, it has not been thoroughly tested for routine use. Examine results closely and interpret them with caution.

If you encounter unexpected behaviour or unusual results, please report them by opening an issue on this repository.

## Usage

pyfastGEAR relies on `python >= 3.10`, `numpy` and `scipy`. Make sure these libraries are installed and accessible to python.

Run with `python3 pyfastGEAR-runner.py --output-dir <RESULTS_FOLDER> <INPUT_FASTA> <OUTPUT_PICKLE>`. `<OUTPUT_PICKLE>` is analogous to the .mat output file of fastGEAR and is not particularly useful, provide `--output-dir` to get more useful results. Specifications can be provided with a fastGEAR input specifications file, using `--input-specs-file`, or manually with command line flags (see `--help`). 

## Citation

If you use fastGEAR or pyfastGEAR in your work, please cite the original publication:

> Mostowy, R., Croucher, N. J., Andam, C. P., Corander, J., Hanage, W. P., & Marttinen, P. (2017). Analysis of recent and ancestral recombination reveals high-resolution population structure in *Streptococcus pneumoniae*. *Molecular Biology and Evolution*, 34(5), 1167–1182.

## Disclaimer

This software is provided **as is**, without warranty of any kind, express or implied.

The authors of pyfastGEAR assume no responsibility for the correctness of results obtained using this software. Users are solely responsible for validating and interpreting any analyses performed with pyfastGEAR.

