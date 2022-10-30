# PyDTINet
PyDTINet is a Python implementation of DTINet, a network integration approach for drug-target interaction prediction. This Python implementation is provided for the convenience of users who want to use DTINet with Python in their own research. For the original implementation (written in MATLAB) associated with the publication that was used to generate the results in the paper, please see the original [DTINet repo](https://github.com/luoyunan/DTINet). More details about the algorithm can be found in our Nature Communications [paper](https://www.nature.com/articles/s41467-017-00680-8).

## Requirements
PyDTINet currently only supports Python 2 due to the dependency on the [IMC](http://bigdata.ices.utexas.edu/software/inductive-matrix-completion/) library used to perform the matrix completion step (we welcome contributions to support Python 3! See [feature request](#feature-requests-of-python-3-support) below). You can create a conda environment with the required dependencies by running the following command:

    conda env create -n dtinet -f environment.yml
    conda activate dtinet

## Usage
1. Install the IMC library.

        cd lib
        bash install_imc.sh
        cd ..
2. Unzip the data files.
        
        unzip data.zip
3. Run a quick demo of the DTINet algorithm.

        cd example
        python demo.py
    Due to a random split of the train/test data, the results may vary slightly from the results produced by the original MATLAB implementation. However, the results should be similar enough to demonstrate the basic functionality of the algorithm.

## Feature requests of Python 3 support
If you would like to contribute to this project by porting the code to Python 3, please feel free to submit a pull request. We will be happy to review it and merge it into the master branch. The major change would be adapting the file that implements the Python bindings for the IMC library to work with Python 3 (see `lib/leml-imf/src/python/train_mf.cpp` after unzipping the `leml-imf-src.zip` file).

## Citation
> Luo, Y., Zhao, X., Zhou, J., Yang, J., Zhang, Y., Kuang, W., Peng, J., Chen, L. & Zeng, J. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. *Nature Communications* **8**, (2017).

    @article{Luo2017,
      author = {Yunan Luo and Xinbin Zhao and Jingtian Zhou and Jinglin Yang and Yanqing Zhang and Wenhua Kuang and Jian Peng and Ligong Chen and Jianyang Zeng},
      title = {A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information},
      doi = {10.1038/s41467-017-00680-8},
      url = {https://doi.org/10.1038/s41467-017-00680-8},
      year  = {2017},
      month = {sep},
      publisher = {Springer Nature},
      volume = {8},
      number = {1},
      journal = {Nature Communications}
    }

## Contacts
Please submit GitHub issues or contact Yunan Luo (luoyunan[at]gmail[dot]com) for any questions related to the source code.