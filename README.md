<h1>IR-pose-classification</h1>
<h2 class="requirements">Requirements</h2>
<ul>
    <li><a href="https://github.com/rbgirshick/yacs">YACS</a> (Yet Another Configuration System)</li>
    <li><a href="https://pytorch.org/">Torch</a> </li>
    <li><a href="https://opencv.org/">Opencv-python</a> </li>
</ul>

<h2>Installation</h2>

- Clone this repository:

```bash
git clone https://github.com/ngocphucck/IR-pose-classification
cd IR-pose-classification
```

- Install dependencies: Please type the command `pip install -r requirements.txt`


<h2>Implementations</h2>

- Firstly, you have to prepare your annotations. I recommend that you organize your labelling file into 2 files: `train.json` and `val.json` with the form `image_path: label`
- After that, you can define some parameters in your method. There are 2 options for you to do that:
    - Change parameters in `defaults.py`
    - Another way is to be more flexible. You'll create YAML configuration files; typically, you'll make one for each experiment. But, when actually implementing, you need to merge this `.yaml` file with `defaults.py.` The following code makes this action:
    
    ```python
    cfg = get_cfg('path_to_file')
    cfg.merge_from_file("experiment.yaml")
    cfg.freeze()
    ```
- Train a model: 
```bash
cd tools
python train.py
```

<h2>Components</h2>
<table align="center">
<tr valign="bottom" align="center">
    <td>Backbones</td>
    <td>Loss functions</td>
    <td>Optimizers</td>
    <td>Augmentations</td>
</tr>
<tr valign="top">
    <td>
    <ul>
        <li>Efficientnet</li>
        <li>ConvNeXt</li>
    </ul>
    </td>
    <td>
    <ul>
        <li>Cross Entropy</li>
    </ul>
    </td>
    <td>
    <ul>
        <li>Adam</li>
    </ul>
    </td>
    <td>
    <ul>
        <li>RandAug</li>
    </ul>
    </td>
</tr>
</table>

<h2>Future works</h2>

- [ ] Multiple extractors 
- [ ] Data augmentations 
- [ ] Multiple loss functions 
