#### HOWTO

* Create as usual a new `env`, I will name it cellpose, but it can be anything :
  * `conda create --name cellpose python=3.8`
* Activate the new `env`: `conda activate cellpose` and then install pytorch:
  
  `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia `

You **DO NOT** have to download/install CUDA etc from nvidia and go through all these painful steps manually. Pytorch comes 
    with cuda binaries already baked-in, you only have to run the command above. Once you do that, verify that pytorch (on the gpu) works fine:

```
    (cellpose) dimitris@egon:~$ python
    Python 3.8.18 (default, Sep 11 2023, 13:40:15) 
    [GCC 11.2.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    >>> import torch
    >>> torch.version.cuda  
    '12.1'
    >>> torch.cuda.is_available()  
    True
    >>> exit()
```

* Install cellpose: `pip install https://github.com/acycliq/cellpose/raw/main/dist/cellpose-0.0.0-py3-none-any.whl` 

    Run the commands below to verify you are running version `0.0.0` and that it runs on the gpu:

```
    (cellpose) dimitris@egon:~/dev/python/cellpose$ python
    Python 3.8.18 (default, Sep 11 2023, 13:40:15) 
    [GCC 11.2.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    >>> import cellpose
    >>> cellpose.__version__
    '0.0.0'
    >>> from cellpose import models, core
    >>> use_GPU = core.use_gpu()
    >>> print('>>> GPU activated? %d'%use_GPU)
    >>> GPU activated? 1
    >>> 
    >>> 
    >>> exit()
```
You can now use the software in exactly the same manner as the official release.

#### NOTES
I wouldnt remove the official release of cellpose straight away. I would keep both on my system (in different envs) and run 2 or 3 examples on both. 
They should yield the same results.  

I would also keep using the official release until it starts becoming too slow or problematic to work with.
If these problems are memory-related and happen during the stitching, then switching to the code you just installed following the instructions 
above might prove helpful to you.

The enhancement implemented here does not require the gpu, but makes more sense to set up cellpose on the gpu anyway.

