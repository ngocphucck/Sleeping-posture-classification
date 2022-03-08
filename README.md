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

<h2>Future works</h2>
<p>
&#x2610; Multiple extractors <br>
&#x2610; Data augmentations <br>
&#x2610; Multiple loss functions <br>
</p>

<h2 class="citations">Citations</h2>
<pre>
@INPROCEEDINGS{9585289,
  author={Doan, Ngoc Phu and Pham, Nguyen Duc Anh and Pham, Hung Manh and Nguyen, Huu Trung and Nguyen, Thuy Anh and Nguyen, Huy Hoang},
  booktitle={2021 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={Real-time Sleeping Posture Recognition For Smart Hospital Beds}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/MAPR53640.2021.9585289}}
</pre>
