Oracle Character Recognition by Nearest Neighbor Classification with Deep Metric Learning
===================================================================================
It is a classification method based on deep metric learning. We use a convolutional neural network to map character images to an Euclidean space where the distance between different samples can measure their similarities such that classification can be performed by the Nearest Neighbor rule. Because new categories are still being discovered in reality, our model enables the rejection of unseen categories and the configuration of new categories.

## structure of dataset

```
--oracle_source
  --oracle_source_img
    --bun_xxt_hard
    --gbk_bronze_lst_seal
    --oracle_54081
    --other_font
  --oracle_source_seq
--oracle_fs
  --img
    --oracle_200_1_shot
      --test
      --train
    --oracle_200_3_shot
      --test
      --train
    --oracle_200_5_shot
      --test
      --train
  --seq
    --oracle_200_1_shot
    --oracle_200_3_shot
    --oracle_200_5_shot
```

## train 

ResNet18：
```bash
python train_base.py --gpu 0 --transfer 0
```

DANN：
```bash
python train_base.py --gpu 0 --transfer 1 --method DANN
```
