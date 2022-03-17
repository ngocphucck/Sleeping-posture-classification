<h1>IR-pose-classification</h1>
<h2 class="requirements">Requirements</h2>
<ul>
    <li><a href="https://github.com/rbgirshick/yacs">YACS</a> (Yet Another Configuration System)</li>
    <li><a href="https://pytorch.org/">Torch</a> </li>
    <li><a href="https://opencv.org/">Opencv-python</a> </li>
</ul>
<h2>Implementations</h2>
Firstly, you have to prepare your annotations in two files: train.json and val.json (with a form <i>image_path: label</i>), and
save them in data directory. After that, change your <b>data_paths</b> in defaults configuration. Run the following scripts
for setup and training steps.

<pre>$ pip install -r requirements.txt
$ cd tools
$ python train.py
</pre>

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
    <ul>
        <li>RandAug</li>
    </ul>
    <td>
        
    </td>
</tr>
</table>

<h2>Future works</h2>
<p>
&#x2610; Multiple extractors <br>
&#x2610; Data augmentations <br>
&#x2610; Multiple loss functions <br>
</p>
