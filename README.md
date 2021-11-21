## 1.文件结构
.
├── config  
│   ├── resnet101_rp2k_config_ascend.yaml  
│   ├── resnet101_rp2k_config_gpu.yaml  
│   ├── resnet18_rp2k_config_ascend.  
│   ├── resnet18_rp2k_config_gpu.yaml  
│   ├── resnet50_rp2k_config_ascend.  
│   └── resnet_benchmark_GPU.yaml  
├── eval.py  
├── GenerateDataset.py  
├── LICENSE  
├── README.md  
├── RP2K_rp2k_dataset  
│   ├── test  
│   └── train  
├── scripts  
│   ├── run_eval_gpu.sh  
│   └── run_train_gpu.sh  
├── src  
│   ├── CrossEntropySmooth.py  
│   ├── dataset_infer.py  
│   ├── dataset.py  
│   ├── eval_callback.py  
│   ├── lr_generator.py  
│   ├── metric.py  
│   ├── model_utils  
│   ├── momentum.py  
│   ├── resnet_gpu_benchmark.py  
│   └── resnet.py  
└── train.py  
config目录下包含了不同模型以及不同运行环境的运行配置文件，详细配置请看注释。  
scripts目录下包含了运行模型的参考运行脚本  
src目录下包含了模型的源码文件  
RP2K_rp2k_dataset包含模型运行所需数据集,结构如下：  
.  
├── test  
│   ├── RP2K_test.mindrecord  
│   └── RP2K_test.mindrecord.db  
└── train  
│  ├── RP2K_train.mindrecord  
│  └── RP2K_train.mindrecord.db  

Tip:注意需要手动添加原始数据集并重命名到原始数据集为RP2K_rp2k_dataset1，在根目录下下运行  
sh scripts/generate_dataset.sh  
即可生成RP2K_rp2k_dataset数据集  
## 2.训练模型
### 2.1 GPU环境
可直接运行:
sh scripts/run_train_gpu.sh  
Tip：由于代码会写入日志以及checkpoint文件到/cache目录下，所以需要在运行环境下新建/cache目录，并给予访问权限。  
sudo mkdir /cache  
chmod 777 /cache

### 2.2 Ascend环境
将数据集和源代码上传到OBS中，在pycharm modelarts kit中创建训练作业(老版本训练作业)，OBS_PATH任意即可，用于保存模型输出checkpoint以及日志文件；DATA_PATH_IN_OBS填写数据集路径,eg:/test_bulket/DATASET/,dataset下级目录包含train和test文件夹。运行文件选择train.py,注意需要添加命令行参数，--config_path config/resnet101_rp2k_config_ascend.yaml    

## 3.验证模型
### 3.1 GPU环境
可直接运行:
sh scripts/run_eval_gpu.sh  
代码会自动到/cache//cache/train/checkpoint目录下寻找最佳ckpt，如果需要自定义ckpt路径，则需要指定对应config文件中的checkpoint_file_path参数。

### 3.2 Ascend环境
#### 3.2.1 相同训练作业  
如果在相同训练作业中运行train和eval任务，则在训练完成后，采用默认的checkpointpath即可完成验证。  
#### 3.2.2 不同训练作业
如果训练任务和验证任务在不同训练作业中，则需要在训练完成后，将得到的best_acc.ckpt保存到obs桶中传入训练任务，然后手动指定chekpoint_path。



