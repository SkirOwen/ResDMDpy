# ResDMDpy: python Residual Dynamic Mode Decomposition

![Static Badge](https://img.shields.io/badge/python->3.11-blue?logo=python)
![GitHub](https://img.shields.io/github/license/SkirOwen/ResDMDpy)

**DISCLAIMER: This an early version and not everything has been ported to python**

This is a `python` translation of https://github.com/MColbrook/Measure-preserving-Extended-Dynamic-Mode-Decomposition.


## Installation


## Use
### Examples
Use the `-e` or `--example` flag to run the examples.  
For now, only `cylinder` is supported.  

#### Cylinder flow

`-m` for the modes to show.  
`-p` for showing the plots (note: plots are currently blocking).

For instance, for modes `1, 2, 3, 4, 5` with plotting.
```shell
python -m rdp -e cylinder -m 1 2 3 4 5 -p
```


## Reference
https://arxiv.org/pdf/2209.02244.pdf