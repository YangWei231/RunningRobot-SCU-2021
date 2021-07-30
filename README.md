# RunningRobot-SCU-2021
The name of our team is Avator. We are SCUers.

Avato_main.py为我们的核心代码

Actionlib.lua为机器人的动作控制文件，里面各数字代表的是相应舵机的刚度或者旋转角度。

lua_to_src.py为我们自己设计的文件转换程序，src为上位机软件中导入动作时的文件格式。lua为机器人内的舵机控制程序。两者有些许区别。

颜色采集文件.py为我们采集到的各颜色在hsv空间上的区间。

RGB2HSV_Sampling_新添功能.py为采集颜色的hsv空间的程序，是我们在比赛方提供的RGB2HSV_sampling.py的基础上的升级版，新添了图片保存功能和数值比较功能。

hsv图片调试.py和hsv流视频调试.py是我们设计的验证颜色区间是否正确的程序。在实际应用中，hsv流视频调试更有用，将其和RGB2HSV_Sampling_新添功能.py程序结合起来使用才能不断精确color_range的值，让机器人的识别更加准确。
