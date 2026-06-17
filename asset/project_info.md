dataset:
iSAID：https://captain-whu.github.io/iSAID/
COCO
VOC
ISIC
GBMSD
Kvasir

为每个dataset创建一个文件夹，下面创建reference_images、reference_masks、target_images、target_masks  4个.txt

说明每个dataset的结构
 datasetname: 一层
 ---classname 一层(类别名称)
 /reference_images #用于few-shot参考
 /reference_masks
 /target_images #用于测试
 /target_masks

utils:
1.读取extract_features.py，train_from_features_multiclass.py
2.将可以复用的函数，按函数意义分不同的.py文件存放。并在项目根目录生成使用utils版本的extract和train的代码。
3.所有编写的代码注意路径用相对路径
4.所有基模型的路径为根目录下
5.需要额外的健壮性工具代码，完成在train,推理过程中的一些模型的检查并终端抛出报错之类的。

脚本 /scripts:
根目录存放下载脚本
1.下载数据集的脚本
2.下载dino模型的脚本
3.下载clip模型的脚本


注意:
extract和train的代码需要选择args读取参数/或者使用./configs/xx.yaml的结构来完成最终版本。
设计的一些参数为:dino版本、dino2/3需要有一个uitils的函数对应不同版本的调用的接口（默认dino3 vithplus）、图像的size尺寸（默认512）,投影层的dim默认512,clip的版本(默认vitb)，bath_size,reference_images下的训练参数(默认10) sam1/2，训练的loss的损失项的权重（默认clip1.其余为0.5）,MAX_K的选择
 
clip的读取路径参考当前的写法，但是是相对路径../clip/clipvitb/...
dino2和dino3的模型路径默认为dinov2 dinov3下的download_models文件夹下

训练生成的投影层在项目根目录生成weights文件夹，文件夹下读取本次训练的投影层的数据集名然后在weights下生成数据集_几张训练图片_dinoxx_clipxx.pth

根目录推理代码：
单张图片的推理代码，参考testMedical.py,test.py，test2.py进行改写和生成。
需要图片路径，一些参数:图像resize,投影层位置（默认使用一个叫./weights/pretrain_coco.pth），投影层dim,输入的文本，输出的分割mask的位置(默认根目录./output)，dino的模型，迭代的次数，文本图像的相似度阈值，迭代的相似度阈值。

这里train和test我觉得可以都用一个config.yaml配置来完成？把一次训练和测试的相关参数都写在一个yaml里面然后读取yaml？你看看怎么是规范合理的?不一定用args的配置，因为我们的参数太多了。如果选用yaml的方案则可以参考./configs/xxx.yaml的结构设计进行读取来提取特征，训练和测试，参考.claude\skills\paper-project-structurer\resources\configs\dinov3_vitb_mlp_infonce.yaml 配置文件的具体参数格式依据我们项目的提取、训练和推理和测试需要来完成


一些可能用到的：
hubconf:dinov2\hubconf.py 可以根据dino2/3的加载需要进行修改，但是最好不改动此文件，而是写在utils下的文件中。
根目录的hubconf.py是改写过的版本，你可以参考这部分的模型加载方式，也可以考虑把这部分移动到uitls文件夹下去存在一个文件中，按照规范做即可。

模型说明：
理论上我们的项目可以使用任何的dino2/dino3的预训练模型，所以在参考testMedical.py进行相关代买生成的时候，我们代码中有关模型的部分最好可以是任意的dino2,dino3的模型名字（可以默认dino3vithplus，不需要全部列出来），而模型的路径则是固定在某个目录下即可。


最终说明：
1.本次任务参考的代码为根路径下的extract_features.py，extract_images_features_dino2.py，hubconf.py，test.py，test2.py，testMedical.py，train_from_features_multiclass.py，train_from_features_multiclass_Weight.py，utils.py
2.生成代码的时候需要互相依赖，注意调用代码的层级需要根据你最终的修改来变化
3.所有代码都依据项目结构使用相对路径。
4.最终根目录下需要生成一GEN开头标识的修改后的训练代码，修改后的单张推理代码。