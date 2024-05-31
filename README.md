# WATT

Official repository of the paper "WATT: Weight Average Test-Time Adaption of CLIP"
This work was greatly inspired by the code in [CLIPArTT](https://github.com/dosowiechi/CLIPArTT.git).

We propose a novel method to achieve Test Time Adaption on VLMs using weight averaging.
![Diagram](https://github.com/dosowiechi/WATT/blob/main/WATT.png)

To run the adaptation, you have to run `adapt.py`. You can choose between both method proposed in the article by putting `--method parallel` to have WATT-P or `--method sequential` to have WATT-S. You can also choose to have the text embedding average during the testing with `--validate-text-features`.


## Citation

If you found this repository, or its related paper useful for your research, you can cite this work as:

```
@inproceedings{}
```
