'''
Compile IMC source code and build *.so file for python
Author: Yunan Luo
'''
unzip leml-imf-src.zip
cd leml-imf
make -C src lib
make -C src/python
cp src/python/__train_mf.so ../../DTINet/
cd ..
rm -r leml-imf 


