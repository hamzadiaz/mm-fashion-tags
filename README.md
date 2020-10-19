### Short Start

Install libraries:

    conda install pytorch torchvision cpuonly -c pytorch
    pip install mmcv-full==latest+torch1.6.0+cpu -f https://download.openmmlab.com/mmcv/dist/index.html
    conda install numpy

---

Create folder names `models` and paste models in it.

---

Use `--coarse` and `--fine` params to select inference mode.
You can use both.

    eg.
    python main.py --input-path [path/of/directory | path/of/single/image] --fine --coarse
    
    python main.py --input-path images/ --fine --coarse
    python main.py --input-path images/demo3.jpg --fine
    python main.py --input-path images/demo1.jpg --coarse
	git clone https://github.com/open-mmlab/mmfashion
	cd mmfashion
	python setup.py install
	

# What is the purpose of the program

It's simplified version of deep-fashion model inference. It's only for inference, there is no training in the script.
If you want to train, use whole original script.
- https://github.com/open-mmlab/mmfashion

### How to predict category ?

- Category prediction is in `--fine`. You need to activate `--fine` prediction. There is 50 category in total.

### How to predict attributes ?

There are two attribute prediction type in program.
- First `--coarse` mode which has 1000 attributes in total,
- Second one is `--fine` mode. Trained more preciesly but has less attribute type against to `-- coarse`. 26 Attribute In `--fine`model total

### Can I use both mode together ? 

Yes you can, and both results will be printed

### How can I select input images ?

Just enter image directory to `--input-path` or just pass only one image path. Both way will work

### What are the labels ?

All labels in `labels` directory