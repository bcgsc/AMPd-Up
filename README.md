# AMPd-Up

AMPd-Up is recurrent neural network based tool for _de novo_ antimicrobial peptide sequence generation.

<p align="center">
	<img src="AMPd-Up.png">
</p>

### Dependencies

* Python 3.6
* PyTorch 1.7.1
* Numpy
* Pandas
* Biopython

### Datasets

The training set (antibacterial sequences only) and known AMP sequences for sequence novelty analysis are stored in the `data` folder.
* Training set: `APD3_ABP_20190320.fa`
* Known AMP sequences: `APD3_20220711.fa` + `DADP_mature_AMP_20181206.fa`

### Pre-trained models

The 1,000 model instances used to generate the peptide sequences presented in the publication can be accessed through the [Zenodo repository](https://doi.org/10.5281/zenodo.7905591). Users can either choose to use the pre-trained models or train their own models for sequence generation.


### Sequence generation
Usage: `AMPd-Up [-h] [-fm FROM_MODEL] -n NUM_SEQ [-sm SAVE_MODEL] [-od OUT_DIR] [-of {fasta,tsv}]`
```
optional arguments:
  -h, --help            Show this help message and exit
  -fm FROM_MODEL, --from_model FROM_MODEL
                        Directory of the existing models; only specify this
                        argument if you want to sample from existing models
                        (optional)
  -n NUM_SEQ, --num_seq NUM_SEQ
                        Number of sequences to sample
  -sm SAVE_MODEL, --save_model SAVE_MODEL
                        Prefix of the models if you want to save them; only
                        specify this argument if you want to sample by
                        training new models (optional)
  -od OUT_DIR, --out_dir OUT_DIR
                        Output directory (optional)
  -of {fasta,tsv}, --out_format {fasta,tsv}
                        Output format, fasta or tsv (tsv by default, optional)
```
Examples:
1) Sample sequences by training new models: `AMPd-Up -n 100`
2) Sample sequences from existing models: `AMPd-Up -fm ../models/ -n 100`

### Author

Chenkai Li (cli@bcgsc.ca)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact us.
