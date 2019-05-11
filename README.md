# Human & Anime face conversion

## Requirements  
 * Face datasets for human and anime  
   * Human face dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset)
   * Anime face dataset: [Danbooru2018](https://www.gwern.net/Danbooru2018), crop the faces using [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)
   * update `data_root` in `configs/human2anime.yaml`. (`/path/to/data_root/` should contain two folders: `anime` and `human`, which contain all face images of the two classes correspondingly)
 * Environment setup (recommend conda)  
   * python=3.6, cuda=9.0  
   * `conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`  
   * `conda install -y -c anaconda pip`  
   * `conda install -y -c anaconda pyyaml`  
   * `pip install tensorboard tensorboardX`
   * `conda install -c 1adrianb face_alignment`


## Code References
  * [NVlabds/MUNIT](https://github.com/NVlabs/MUNIT)  
  * [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)
