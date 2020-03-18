auto color paint for comic using pix2pix and WDSR(keras),based on user:wmylxmj job
https://github.com/wmylxmj/Pix2Pix-Keras
https://github.com/wmylxmj/Anime-Super-Resolution
感谢大佬的开源项目
漫画、动漫壁纸、本子自动上色，目前先训练了本子的自动上色，时长12个钟，模型效果还行，起码比黑白的好看2333
训练example show:
左边灰度(gray) 中间预测(predict) 右边原图(original)
![example1](/images/1.PNG)
![example2](/images/2.PNG)
![example3](/images/3.PNG)

用训练好后的模型对黑白本子上色效果如下(gray img non-resize predict)：
原图original：![黑白本子](/images/9.jpg)
预测predict：![自动上色后](/images/combine_9.jpg)

how to use:
<br/>1.train model 训练自己需要的上色模型:just see wmylxmj job：https://github.com/wmylxmj/Pix2Pix-Keras
<br/>2.get colored_img 得到上色后图片：by running code predict.py 

项目目录(Directory)：
<br/>./weights:存放训练后的上色模型以及SR模型，包含wdsr-b-32-x4.h5、discriminator_weights.h5(可省略)、generator_weights.h5
<br/>./datasets/OriginalImages:存放用于训练的彩色图片
<br/>pix2pix.py:pix2pix model file
<br/>WDSR.py:wdsr model file
<br/>utils.py: settings of loading data 设置加载数据的方法等
<br/>prepare.py: pre-step before you start train 训练前的预处理数据
<br/>predict.py: using trained model get colored img 得到上色后图片

项目API使用:
<br/>使用Mo平台部署在线测试(Deploy test online):支持javascript、curl、python
<br/>流程：调用API初始化图片，返回上色后图片的base64字符串，使用函数转成img另存本地
![step1](/images/test_step1.PNG)
![step2](/images/test_step2.PNG)
