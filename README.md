# distributed_tensorflow
Sample Implementation of Distributed TensorFlow

This should work for TF > 1.4.

It implements either synchronous and asynchronous parameter updates.

The "data" is just a simple line with random noise. You can change the slope and intercept of the line by editing those variables at the top of the test_dist.py file.

```
slope = 5
intercept = 13
```

To run the script:
1. First you'll need to install a TensorFlow conda environment on all of your machines.
+ Install [Miniconda](https://conda.io/miniconda.html)
+ `conda create -n tf -c intel python=2 pip numpy`
+ `source activate tf`
+ `pip install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl`
2. Edit test_dist.py to specify which machines will be parameter servers and which will be worker nodes.
```
ps_hosts = ["10.100.68.245"]
ps_ports = ["2222"]
worker_hosts = ["10.100.68.193","10.100.68.183"] #,"10.100.68.185","10.100.68.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]
```
3. Run `python test_dist.py --issync=1` on every machine. You can specify the IP address (--ip) if you want to use OPA instead of the usual IPv4/6 address.

